"""
tests/test_scorer.py
--------------------
Unit tests for the ResumeIQ scoring engine.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.skill_extractor import (
    extract_skills_from_text,
    extract_years_of_experience,
    extract_education_level,
    full_extraction,
)
from src.preprocessor import clean_text, preprocess
from src.scorer import ResumeScorer, ScoringConfig, skill_match_score, experience_score
from src.ranker import rank_candidates, generate_explanation, summary_report


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessor Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_clean_removes_urls(self):
        text = "Visit https://example.com for more info"
        cleaned = clean_text(text)
        assert "http" not in cleaned
        assert "example" not in cleaned

    def test_clean_removes_emails(self):
        text = "Contact me at john@example.com"
        cleaned = clean_text(text)
        assert "@" not in cleaned

    def test_clean_lowercases(self):
        text = "Python TensorFlow DEEP LEARNING"
        cleaned = clean_text(text)
        assert cleaned == cleaned.lower()

    def test_clean_normalizes_whitespace(self):
        text = "hello   world\n\n\nfoo"
        cleaned = clean_text(text)
        assert "  " not in cleaned

    def test_preprocess_returns_all_keys(self):
        result = preprocess("I am a Python developer with TensorFlow experience.")
        assert "raw" in result
        assert "cleaned" in result
        assert "lemmatized" in result
        assert "tokens" in result

    def test_preprocess_empty_string(self):
        result = preprocess("")
        assert result["cleaned"] == ""
        assert result["tokens"] == []


# ──────────────────────────────────────────────────────────────────────────────
# Skill Extractor Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSkillExtractor:
    def test_extract_python(self):
        skills = extract_skills_from_text("Experienced Python developer")
        assert "python" in skills

    def test_extract_multiple_skills(self):
        text = "Skills: Python, TensorFlow, Docker, SQL, Git"
        skills = extract_skills_from_text(text)
        assert "python" in skills
        assert "tensorflow" in skills
        assert "docker" in skills
        assert "sql" in skills
        assert "git" in skills

    def test_extract_aliases(self):
        # "sklearn" should map to "scikit-learn"
        skills = extract_skills_from_text("I use sklearn for modelling")
        assert "scikit-learn" in skills

    def test_extract_pyspark_maps_to_spark(self):
        skills = extract_skills_from_text("Built ETL pipelines with PySpark")
        assert "spark" in skills

    def test_extract_years_basic(self):
        years = extract_years_of_experience("5 years of experience in ML")
        assert years == 5.0

    def test_extract_years_plus(self):
        years = extract_years_of_experience("3+ years experience")
        assert years == 3.0

    def test_extract_years_range(self):
        years = extract_years_of_experience("2-4 years of experience")
        assert years == 3.0  # average

    def test_extract_years_none(self):
        years = extract_years_of_experience("No experience information provided")
        assert years is None

    def test_education_phd(self):
        edu = extract_education_level("PhD in Computer Science from MIT")
        assert edu == "phd"

    def test_education_masters(self):
        edu = extract_education_level("M.S. in Data Science, Stanford")
        assert edu == "masters"

    def test_education_bachelors(self):
        edu = extract_education_level("Bachelor of Science in Statistics")
        assert edu == "bachelors"

    def test_education_none(self):
        edu = extract_education_level("Software developer with 5 years experience")
        assert edu == "none"

    def test_full_extraction_returns_all_keys(self):
        text = "Python developer, 3 years experience, Bachelor's in CS"
        result = full_extraction(text)
        assert "skills" in result
        assert "years_experience" in result
        assert "education_level" in result
        assert "contact" in result
        assert "skill_count" in result


# ──────────────────────────────────────────────────────────────────────────────
# Scorer Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestScorer:
    def test_config_valid_weights(self):
        config = ScoringConfig(semantic_weight=0.45, skill_weight=0.40, experience_weight=0.15)
        config.validate()  # Should not raise

    def test_config_invalid_weights(self):
        config = ScoringConfig(semantic_weight=0.5, skill_weight=0.5, experience_weight=0.5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_skill_match_perfect(self):
        result = skill_match_score(
            resume_skills={"python", "tensorflow", "docker"},
            required_skills={"python", "tensorflow", "docker"},
        )
        assert result["score"] == 1.0
        assert result["missing"] == set()

    def test_skill_match_none(self):
        result = skill_match_score(
            resume_skills={"excel", "powerpoint"},
            required_skills={"python", "tensorflow", "docker"},
        )
        assert result["score"] == 0.0
        assert result["matched"] == set()

    def test_skill_match_partial(self):
        result = skill_match_score(
            resume_skills={"python", "excel"},
            required_skills={"python", "tensorflow", "docker"},
        )
        assert 0.0 < result["score"] < 1.0
        assert "python" in result["matched"]
        assert "tensorflow" in result["missing"]

    def test_skill_match_empty_required(self):
        result = skill_match_score(
            resume_skills={"python"},
            required_skills=set(),
        )
        assert result["score"] == 0.5  # neutral when no requirements

    def test_experience_score_phd(self):
        score = experience_score("phd", 10.0)
        assert score > 0.8

    def test_experience_score_none(self):
        score = experience_score("none", None)
        assert 0.0 <= score <= 1.0

    def test_experience_score_vs_requirement(self):
        # 5 years when 5 required → high score
        score_met = experience_score("bachelors", 5.0, required_years=5.0)
        # 1 year when 5 required → lower score
        score_unmet = experience_score("bachelors", 1.0, required_years=5.0)
        assert score_met > score_unmet

    def test_scorer_returns_all_keys(self):
        scorer = ResumeScorer()
        resume_data = {
            "lemmatized": "python developer tensorflow experience",
            "cleaned": "python developer tensorflow experience",
            "skills": {"python", "tensorflow"},
            "education_level": "bachelors",
            "years_experience": 3.0,
            "contact": {},
        }
        jd_data = {
            "lemmatized": "python machine learning tensorflow aws",
            "cleaned": "python machine learning tensorflow aws",
            "skills": {"python", "tensorflow", "aws"},
        }
        result = scorer.score_resume(resume_data, jd_data)
        assert "final_score" in result
        assert "tier" in result
        assert "signals" in result
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert 0.0 <= result["final_score"] <= 1.0

    def test_scorer_high_score_for_perfect_match(self):
        scorer = ResumeScorer()
        text = "python tensorflow deep learning machine learning nlp docker kubernetes aws mlops"
        resume_data = {
            "lemmatized": text,
            "cleaned": text,
            "skills": {"python", "tensorflow", "deep learning", "machine learning",
                        "natural language processing", "docker", "kubernetes", "aws"},
            "education_level": "masters",
            "years_experience": 5.0,
            "contact": {},
        }
        jd_data = {
            "lemmatized": text,
            "cleaned": text,
            "skills": {"python", "tensorflow", "deep learning", "machine learning",
                        "natural language processing", "docker", "kubernetes", "aws"},
        }
        result = scorer.score_resume(resume_data, jd_data)
        assert result["final_score"] > 0.7

    def test_scorer_low_score_for_mismatch(self):
        scorer = ResumeScorer()
        resume_data = {
            "lemmatized": "excel powerpoint word office administration scheduling",
            "cleaned": "excel powerpoint word office administration scheduling",
            "skills": set(),
            "education_level": "none",
            "years_experience": None,
            "contact": {},
        }
        jd_data = {
            "lemmatized": "python tensorflow deep learning computer vision neural networks",
            "cleaned": "python tensorflow deep learning computer vision neural networks",
            "skills": {"python", "tensorflow", "deep learning", "computer vision"},
        }
        result = scorer.score_resume(resume_data, jd_data)
        assert result["final_score"] < 0.5

    def test_batch_scoring_consistent(self):
        scorer = ResumeScorer()
        resume_data = {
            "lemmatized": "python machine learning sklearn",
            "cleaned": "python machine learning sklearn",
            "skills": {"python", "machine learning", "scikit-learn"},
            "education_level": "bachelors",
            "years_experience": 2.0,
            "contact": {},
        }
        jd_data = {
            "lemmatized": "python machine learning deep learning",
            "cleaned": "python machine learning deep learning",
            "skills": {"python", "machine learning", "deep learning"},
        }
        single = scorer.score_resume(resume_data, jd_data)
        batch = scorer.score_batch([resume_data], jd_data)
        # Scores should be very close (not necessarily identical due to tfidf state)
        assert abs(single["final_score"] - batch[0]["final_score"]) < 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Ranker Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRanker:
    def _make_result(self, score, matched, missing):
        return {
            "final_score": score,
            "final_score_pct": f"{score*100:.1f}%",
            "tier": "Good Match",
            "tier_color": "#84cc16",
            "signals": {
                "semantic": {"score": score, "weight": 0.45, "weighted": score * 0.45,
                              "label": "Semantic", "description": ""},
                "skill_match": {"score": score, "weight": 0.40, "weighted": score * 0.40,
                                 "label": "Skill Match", "description": ""},
                "experience": {"score": score, "weight": 0.15, "weighted": score * 0.15,
                                "label": "Experience", "description": ""},
            },
            "matched_skills": matched,
            "missing_skills": missing,
            "bonus_skills": set(),
            "preferred_match": set(),
            "education_level": "bachelors",
            "years_experience": 3.0,
            "contact": {},
        }

    def test_ranking_order(self):
        results = [
            self._make_result(0.60, {"python"}, {"tensorflow"}),
            self._make_result(0.90, {"python", "tensorflow"}, set()),
            self._make_result(0.30, set(), {"python", "tensorflow"}),
        ]
        df = rank_candidates(results, ["B", "A", "C"])
        assert df.iloc[0]["name"] == "A"  # highest score
        assert df.iloc[-1]["name"] == "C"  # lowest score

    def test_ranking_columns(self):
        results = [self._make_result(0.70, {"python"}, {"tensorflow"})]
        df = rank_candidates(results, ["Test"])
        expected_cols = ["name", "final_score", "tier", "matched_skills",
                          "missing_skills", "explanation"]
        for col in expected_cols:
            assert col in df.columns

    def test_explanation_generated(self):
        result = self._make_result(0.85, {"python", "tensorflow"}, {"docker"})
        explanation = generate_explanation(result, "Test Candidate")
        assert "Test Candidate" in explanation
        assert len(explanation) > 100

    def test_summary_report(self):
        results = [
            self._make_result(0.80, {"python"}, set()),
            self._make_result(0.60, {"python"}, {"docker"}),
            self._make_result(0.35, set(), {"python", "docker"}),
        ]
        df = rank_candidates(results, ["A", "B", "C"])
        summary = summary_report(df, "ML Engineer")
        assert summary["total_candidates"] == 3
        assert "average_score" in summary
        assert "strong_matches" in summary

    def test_empty_input(self):
        df = rank_candidates([], [])
        assert df.empty


# ──────────────────────────────────────────────────────────────────────────────
# Integration Test
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_text_input(self):
        """End-to-end test with raw text inputs."""
        from src.pipeline import ResumeIQPipeline

        jd_text = """
        We need a Machine Learning Engineer with:
        - Python, TensorFlow, PyTorch
        - 3+ years of experience
        - Docker, Kubernetes knowledge
        - Bachelor's degree in CS
        """

        resumes = [
            """
            Jane Doe — ML Engineer
            5 years of experience.
            Skills: Python, TensorFlow, PyTorch, Docker, Kubernetes, scikit-learn
            Education: Master's in Computer Science
            """,
            """
            Bob Smith — Data Analyst
            1 year experience.
            Skills: Excel, SQL, Power BI
            Education: Bachelor's in Business
            """,
        ]

        pipeline = ResumeIQPipeline()
        df, summary = pipeline.run(resumes=resumes, job_description=jd_text, job_title="ML Engineer")

        assert len(df) == 2
        assert df.iloc[0]["final_score"] > df.iloc[1]["final_score"]
        assert summary["total_candidates"] == 2
        assert "top_candidate" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
