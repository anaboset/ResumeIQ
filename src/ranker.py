"""
ranker.py
---------
Takes scorer output and produces:
- A ranked DataFrame of candidates
- Human-readable explanations for each candidate
- Summary statistics for recruiter reporting
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explanation Generator
# ---------------------------------------------------------------------------

EDUCATION_LABELS = {
    "phd": "PhD",
    "masters": "Master's degree",
    "bachelors": "Bachelor's degree",
    "associate": "Associate degree",
    "diploma": "Diploma / Certificate",
    "none": "Not specified",
}


def generate_explanation(result: dict, candidate_name: str = "Candidate") -> str:
    """
    Generate a recruiter-friendly explanation for a candidate's score.
    Returns plain English text explaining why the candidate ranked as they did.
    """
    score = result["final_score"]
    tier = result["tier"]
    signals = result["signals"]
    matched = result.get("matched_skills", set())
    missing = result.get("missing_skills", set())
    bonus = result.get("bonus_skills", set())
    edu = result.get("education_level", "none")
    years = result.get("years_experience")

    lines = []

    # Opening
    lines.append(
        f"{candidate_name} scored {score * 100:.1f}% overall and is classified as a "
        f"**{tier}** for this role."
    )

    # Semantic
    sem = signals["semantic"]["score"]
    if sem >= 0.75:
        lines.append(
            "Their resume language closely mirrors the job description, suggesting "
            "strong alignment in terminology, domain, and focus areas."
        )
    elif sem >= 0.50:
        lines.append(
            "There is moderate semantic overlap between their resume and the job description. "
            "The candidate covers several relevant areas but may use different terminology."
        )
    else:
        lines.append(
            "There is limited semantic overlap with the job description. "
            "The candidate's background may be in a different domain or specialty."
        )

    # Skill match
    sk = signals["skill_match"]["score"]
    if matched:
        top_matched = sorted(matched)[:8]
        lines.append(
            f"**Matched skills ({len(matched)}):** {', '.join(top_matched)}"
            + (f" and {len(matched) - 8} more." if len(matched) > 8 else ".")
        )
    if missing:
        top_missing = sorted(missing)[:6]
        lines.append(
            f"**Missing skills ({len(missing)}):** {', '.join(top_missing)}"
            + (f" and {len(missing) - 6} more." if len(missing) > 6 else ".")
            + " These are gaps the candidate would need to address."
        )
    if bonus and len(bonus) > 0:
        top_bonus = sorted(bonus)[:5]
        lines.append(
            f"**Bonus skills:** {', '.join(top_bonus)} — not required, but add value."
        )

    # Experience
    edu_label = EDUCATION_LABELS.get(edu, "Not specified")
    if years:
        lines.append(
            f"**Experience:** {edu_label}, {years:.0f} year(s) of experience indicated."
        )
    else:
        lines.append(
            f"**Education:** {edu_label}. Years of experience not clearly specified."
        )

    # Recommendation
    if score >= 0.75:
        lines.append("✅ **Recommendation:** Strongly consider for interview.")
    elif score >= 0.55:
        lines.append("🔵 **Recommendation:** Good candidate — worth a screening call.")
    elif score >= 0.40:
        lines.append(
            "⚠️ **Recommendation:** Partial match. May be suitable if the team can "
            "invest in skill development."
        )
    else:
        lines.append(
            "❌ **Recommendation:** Weak match for this specific role. "
            "May be better suited for a different position."
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: list[dict],
    candidate_names: Optional[list[str]] = None,
    jd_data: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Rank candidates by final score and return a structured DataFrame.
    
    Args:
        candidates:       list of scorer output dicts
        candidate_names:  optional list of candidate names/filenames
        jd_data:          job description data (used for required skills count)

    Returns:
        DataFrame with rank, scores, skill analysis, and explanations.
    """
    if not candidates:
        return pd.DataFrame()

    if candidate_names is None:
        candidate_names = [f"Candidate {i+1}" for i in range(len(candidates))]

    required_skills = jd_data.get("skills", set()) if jd_data else set()
    total_required = len(required_skills)

    rows = []
    for name, result in zip(candidate_names, candidates):
        signals = result.get("signals", {})
        matched = result.get("matched_skills", set())
        missing = result.get("missing_skills", set())
        bonus = result.get("bonus_skills", set())

        explanation = generate_explanation(result, candidate_name=name)

        rows.append({
            "name": name,
            "final_score": result["final_score"],
            "final_score_pct": result["final_score_pct"],
            "tier": result["tier"],
            "tier_color": result["tier_color"],

            # Signal scores
            "semantic_score": signals.get("semantic", {}).get("score", 0),
            "skill_match_score": signals.get("skill_match", {}).get("score", 0),
            "experience_score": signals.get("experience", {}).get("score", 0),

            # Skill details
            "matched_count": len(matched),
            "missing_count": len(missing),
            "bonus_count": len(bonus),
            "total_required": total_required,
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
            "bonus_skills": sorted(bonus),

            # Profile
            "education_level": result.get("education_level", "none"),
            "years_experience": result.get("years_experience"),
            "email": result.get("contact", {}).get("email"),
            "phone": result.get("contact", {}).get("phone"),

            # Explanation
            "explanation": explanation,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "rank"
    return df


def summary_report(df: pd.DataFrame, job_title: str = "the role") -> dict:
    """
    Generate summary statistics for the recruiter overview panel.
    """
    if df.empty:
        return {}

    tier_counts = df["tier"].value_counts().to_dict()
    avg_score = df["final_score"].mean()
    top_candidate = df.iloc[0]["name"] if not df.empty else "N/A"

    # Skill gap analysis across all candidates
    all_missing = []
    for skills in df["missing_skills"]:
        all_missing.extend(skills)

    from collections import Counter
    common_gaps = Counter(all_missing).most_common(10)

    return {
        "total_candidates": len(df),
        "job_title": job_title,
        "average_score": f"{avg_score * 100:.1f}%",
        "top_candidate": top_candidate,
        "top_score": df.iloc[0]["final_score_pct"] if not df.empty else "N/A",
        "tier_breakdown": tier_counts,
        "common_skill_gaps": common_gaps,
        "strong_matches": int((df["final_score"] >= 0.75).sum()),
        "good_matches": int(((df["final_score"] >= 0.55) & (df["final_score"] < 0.75)).sum()),
        "partial_matches": int(((df["final_score"] >= 0.40) & (df["final_score"] < 0.55)).sum()),
        "weak_matches": int((df["final_score"] < 0.40).sum()),
    }


def export_to_csv(df: pd.DataFrame, path: str) -> None:
    """Export ranked results to CSV (skill sets serialized as pipe-separated strings)."""
    export_df = df.copy()
    for col in ["matched_skills", "missing_skills", "bonus_skills"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: " | ".join(x) if isinstance(x, list) else x
            )
    # Drop explanation column for CSV (too long)
    if "explanation" in export_df.columns:
        export_df = export_df.drop(columns=["explanation"])
    export_df.to_csv(path, index=True)
    logger.info(f"Results exported to {path}")
