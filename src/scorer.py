"""
scorer.py
---------
Multi-signal ensemble scoring engine.

Combines three independent signals:
  1. Semantic Similarity  (45%) — sentence-transformers cosine similarity
  2. Skill Match Score    (40%) — taxonomy-based exact/alias skill overlap
  3. Experience Score     (15%) — education level + years of experience

Each signal is independently normalized to [0, 1] before weighting.
Weights are configurable via the ScoringConfig dataclass.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Scoring weights and parameters. Must sum to 1.0."""
    semantic_weight: float = 0.45
    skill_weight: float = 0.40
    experience_weight: float = 0.15

    # Semantic backend: "sentence_transformers" | "tfidf"
    semantic_backend: str = "sentence_transformers"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # Skill scoring
    required_skills_bonus: float = 1.5   # multiplier for explicitly required skills
    preferred_skills_weight: float = 0.5 # weight of preferred vs required

    # Education level → numeric score
    education_scores: dict = field(default_factory=lambda: {
        "phd": 1.0,
        "masters": 0.85,
        "bachelors": 0.70,
        "associate": 0.50,
        "diploma": 0.35,
        "none": 0.20,
    })

    def validate(self):
        total = self.semantic_weight + self.skill_weight + self.experience_weight
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


DEFAULT_CONFIG = ScoringConfig()


# ---------------------------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------------------------

class SemanticScorer:
    """Computes semantic similarity between resume and JD text."""
    _model_cache = None

    def __init__(self, config: ScoringConfig = DEFAULT_CONFIG):
        self.config = config
        self._tfidf = None
        self._backend = None

    def _load_model(self):
        if SemanticScorer._model_cache is not None:
            self._model = SemanticScorer._model_cache
            return

        if self.config.semantic_backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.config.sentence_transformer_model)
                SemanticScorer._model_cache = model
                self._model = model
                logger.info("Loaded sentence-transformers backend.")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Falling back to TF-IDF. "
                    "Install with: pip install sentence-transformers"
                )
                self._backend = "tfidf"

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        self._load_model()
        if self._backend == "sentence_transformers":
            return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            # TF-IDF fallback
            try:
                return self._tfidf.fit_transform(texts).toarray()
            except Exception:
                self._tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
                return self._tfidf.fit_transform(texts).toarray()

    def score(self, resume_text: str, jd_text: str) -> float:
        """
        Returns cosine similarity between resume and JD in [0, 1].
        """
        if not resume_text or not jd_text:
            return 0.0
        try:
            embeddings = self.encode([resume_text, jd_text])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Semantic scoring failed: {e}")
            return 0.0

    def score_batch(self, resume_texts: list[str], jd_text: str) -> list[float]:
        """
        Score multiple resumes against one JD efficiently.
        Uses single encoding call for all resumes + JD.
        """
        if not resume_texts:
            return []
        self._load_model()
        try:
            all_texts = resume_texts + [jd_text]
            embeddings = self.encode(all_texts)
            jd_emb = embeddings[-1:]
            resume_embs = embeddings[:-1]
            sims = cosine_similarity(resume_embs, jd_emb).flatten()
            return [float(np.clip(s, 0.0, 1.0)) for s in sims]
        except Exception as e:
            logger.error(f"Batch semantic scoring failed: {e}")
            return [0.0] * len(resume_texts)


# ---------------------------------------------------------------------------
# Skill Match Scoring
# ---------------------------------------------------------------------------

def skill_match_score(
    resume_skills: set[str],
    required_skills: set[str],
    preferred_skills: Optional[set[str]] = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Compute skill match score and gap analysis.

    Returns:
        score         : float in [0, 1]
        matched       : skills in both resume and required
        missing       : required skills not in resume
        bonus_skills  : resume skills not required (extras)
        preferred_match: preferred skills found in resume
    """
    preferred_skills = preferred_skills or set()

    if not required_skills:
        return {
            "score": 0.5,
            "matched": set(),
            "missing": set(),
            "bonus_skills": resume_skills,
            "preferred_match": set(),
        }

    # Core required match
    matched = resume_skills & required_skills
    missing = required_skills - resume_skills
    bonus_skills = resume_skills - required_skills - preferred_skills

    required_score = len(matched) / len(required_skills) if required_skills else 0

    # Preferred skills (soft bonus)
    preferred_match = resume_skills & preferred_skills
    preferred_score = (
        len(preferred_match) / len(preferred_skills)
        if preferred_skills else 0
    )

    # Weighted combination of required + preferred
    if preferred_skills:
        skill_score = (
            required_score * (1 - config.preferred_skills_weight)
            + preferred_score * config.preferred_skills_weight
        )
    else:
        skill_score = required_score

    return {
        "score": float(np.clip(skill_score, 0.0, 1.0)),
        "matched": matched,
        "missing": missing,
        "bonus_skills": bonus_skills,
        "preferred_match": preferred_match,
    }


# ---------------------------------------------------------------------------
# Experience Scoring
# ---------------------------------------------------------------------------

def experience_score(
    education_level: str,
    years_experience: Optional[float],
    required_years: Optional[float] = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> float:
    """
    Compute normalized experience score from education + years.
    
    Score breakdown:
      - 60% from education level
      - 40% from years of experience (vs required or vs general benchmark)
    """
    edu_score = config.education_scores.get(education_level, 0.2)

    # Years of experience sub-score
    if years_experience is None:
        exp_score = 0.3  # Unknown → slightly penalized
    elif required_years is not None and required_years > 0:
        # Direct comparison to requirement
        exp_score = min(1.0, years_experience / required_years)
    else:
        # General benchmark: 0–10+ years mapped to 0–1
        exp_score = min(1.0, years_experience / 10.0)

    final = 0.60 * edu_score + 0.40 * exp_score
    return float(np.clip(final, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main Ensemble Scorer
# ---------------------------------------------------------------------------

class ResumeScorer:
    """
    End-to-end ensemble scorer for resume-JD matching.
    
    Usage:
        scorer = ResumeScorer()
        result = scorer.score_resume(resume_data, jd_data)
    """

    def __init__(self, config: ScoringConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        self.semantic = SemanticScorer(self.config)

    def score_resume(
        self,
        resume_data: dict,
        jd_data: dict,
        required_years: Optional[float] = None,
    ) -> dict:
        """
        Score a single resume against a job description.

        Args:
            resume_data: output of skill_extractor.full_extraction() + preprocessor.preprocess()
            jd_data:     same structure for the job description
            required_years: explicitly required years (parsed from JD or overridden)

        Returns:
            Comprehensive scoring dict with final_score, signal scores, and gap analysis.
        """
        # --- Signal 1: Semantic Similarity ---
        sem_score = self.semantic.score(
            resume_data.get("lemmatized", resume_data.get("cleaned", "")),
            jd_data.get("lemmatized", jd_data.get("cleaned", "")),
        )

        # --- Signal 2: Skill Match ---
        skill_result = skill_match_score(
            resume_skills=resume_data.get("skills", set()),
            required_skills=jd_data.get("skills", set()),
            preferred_skills=jd_data.get("preferred_skills", set()),
            config=self.config,
        )

        # --- Signal 3: Experience ---
        exp_score = experience_score(
            education_level=resume_data.get("education_level", "none"),
            years_experience=resume_data.get("years_experience"),
            required_years=required_years or jd_data.get("years_experience"),
            config=self.config,
        )

        # --- Weighted Ensemble ---
        final = (
            self.config.semantic_weight * sem_score
            + self.config.skill_weight * skill_result["score"]
            + self.config.experience_weight * exp_score
        )
        final = float(np.clip(final, 0.0, 1.0))

        # --- Tier Classification ---
        if final >= 0.75:
            tier = "Strong Match"
            tier_color = "#22c55e"
        elif final >= 0.55:
            tier = "Good Match"
            tier_color = "#84cc16"
        elif final >= 0.40:
            tier = "Partial Match"
            tier_color = "#f59e0b"
        else:
            tier = "Weak Match"
            tier_color = "#ef4444"

        return {
            # Core output
            "final_score": final,
            "final_score_pct": f"{final * 100:.1f}%",
            "tier": tier,
            "tier_color": tier_color,

            # Signal breakdown
            "signals": {
                "semantic": {
                    "score": sem_score,
                    "weight": self.config.semantic_weight,
                    "weighted": sem_score * self.config.semantic_weight,
                    "label": "Semantic Similarity",
                    "description": "How closely the resume language matches the job description",
                },
                "skill_match": {
                    "score": skill_result["score"],
                    "weight": self.config.skill_weight,
                    "weighted": skill_result["score"] * self.config.skill_weight,
                    "label": "Skill Match",
                    "description": "Percentage of required skills found in the resume",
                },
                "experience": {
                    "score": exp_score,
                    "weight": self.config.experience_weight,
                    "weighted": exp_score * self.config.experience_weight,
                    "label": "Experience & Education",
                    "description": "Education level and years of experience alignment",
                },
            },

            # Skill gap analysis
            "matched_skills": skill_result["matched"],
            "missing_skills": skill_result["missing"],
            "bonus_skills": skill_result["bonus_skills"],
            "preferred_match": skill_result["preferred_match"],

            # Resume metadata
            "education_level": resume_data.get("education_level", "none"),
            "years_experience": resume_data.get("years_experience"),
            "contact": resume_data.get("contact", {}),
        }

    def score_batch(
        self,
        resume_data_list: list[dict],
        jd_data: dict,
        required_years: Optional[float] = None,
    ) -> list[dict]:
        """
        Score multiple resumes efficiently using batched semantic encoding.
        """
        if not resume_data_list:
            return []

        # Batch semantic scores
        resume_texts = [
            r.get("lemmatized", r.get("cleaned", "")) for r in resume_data_list
        ]
        jd_text = jd_data.get("lemmatized", jd_data.get("cleaned", ""))
        sem_scores = self.semantic.score_batch(resume_texts, jd_text)

        results = []
        for i, (resume_data, sem_score) in enumerate(zip(resume_data_list, sem_scores)):
            # Skill match
            skill_result = skill_match_score(
                resume_skills=resume_data.get("skills", set()),
                required_skills=jd_data.get("skills", set()),
                preferred_skills=jd_data.get("preferred_skills", set()),
                config=self.config,
            )
            # Experience
            exp_score = experience_score(
                education_level=resume_data.get("education_level", "none"),
                years_experience=resume_data.get("years_experience"),
                required_years=required_years or jd_data.get("years_experience"),
                config=self.config,
            )
            # Ensemble
            final = float(np.clip(
                self.config.semantic_weight * sem_score
                + self.config.skill_weight * skill_result["score"]
                + self.config.experience_weight * exp_score,
                0.0, 1.0
            ))

            if final >= 0.75:
                tier, tier_color = "Strong Match", "#22c55e"
            elif final >= 0.55:
                tier, tier_color = "Good Match", "#84cc16"
            elif final >= 0.40:
                tier, tier_color = "Partial Match", "#f59e0b"
            else:
                tier, tier_color = "Weak Match", "#ef4444"

            results.append({
                "final_score": final,
                "final_score_pct": f"{final * 100:.1f}%",
                "tier": tier,
                "tier_color": tier_color,
                "signals": {
                    "semantic": {"score": sem_score, "weight": self.config.semantic_weight,
                                  "weighted": sem_score * self.config.semantic_weight},
                    "skill_match": {"score": skill_result["score"], "weight": self.config.skill_weight,
                                     "weighted": skill_result["score"] * self.config.skill_weight},
                    "experience": {"score": exp_score, "weight": self.config.experience_weight,
                                    "weighted": exp_score * self.config.experience_weight},
                },
                "matched_skills": skill_result["matched"],
                "missing_skills": skill_result["missing"],
                "bonus_skills": skill_result["bonus_skills"],
                "preferred_match": skill_result["preferred_match"],
                "education_level": resume_data.get("education_level", "none"),
                "years_experience": resume_data.get("years_experience"),
                "contact": resume_data.get("contact", {}),
            })

        return results
