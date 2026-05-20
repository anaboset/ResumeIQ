"""
app/dashboard.py
----------------
ResumeIQ Streamlit Dashboard — the main interactive UI.

Run with:
    streamlit run app/dashboard.py
"""

import sys
import os
import io
import time
import tempfile
import atexit
from typing import Tuple
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.scorer import ScoringConfig
from src.skill_extractor import SKILLS_TAXONOMY
from data.demo_data import DEMO_JD, DEMO_RESUMES
from app.ui.styles import CUSTOM_CSS
from app.ui.charts import (
    score_gauge,
    signal_bar_chart,
    comparison_radar,
    tier_donut,
)
from app.utils.files import save_uploaded_file

logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — AI Candidate Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Scoring Config")
    st.caption("Adjust weights for each scoring signal.")

    sem_w = st.slider("Semantic Similarity", 0.0, 1.0, 0.45, 0.05,
                       help="How much the resume language matches the JD")
    sk_w = st.slider("Skill Match", 0.0, 1.0, 0.40, 0.05,
                      help="Percentage of required skills found")
    exp_w = st.slider("Experience & Education", 0.0, 1.0, 0.15, 0.05,
                       help="Education level and years of experience")

    total = sem_w + sk_w + exp_w
    if abs(total - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total:.2f}. They must sum to 1.0.")
        config_valid = False
    else:
        st.success("✅ Weights sum to 1.0")
        config_valid = True

    st.markdown("---")
    st.markdown("### 🧠 Semantic Backend")
    backend = st.selectbox(
        "Model",
        ["sentence_transformers (best)", "tfidf (fast, offline)"],
        help="sentence-transformers requires internet on first run to download model"
    )
    use_st = "sentence_transformers" in backend

    st.markdown("---")
    st.markdown("### 📋 About")
    st.caption(
        "**ResumeIQ** uses a 3-signal ensemble:\n"
        "- Semantic similarity (MiniLM)\n"
        "- Skill taxonomy matching\n"
        "- Education + experience scoring\n\n"
        "Built by a Future Interns ML intern."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="main-title">Resume<span class="accent">IQ</span></div>
    <div class="main-subtitle">
        ML-powered candidate screening · Semantic similarity · Skill gap analysis · Explainable rankings
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Input Section
# ──────────────────────────────────────────────────────────────────────────────

col_jd, col_resume = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown('<div class="section-header">📄 Job Description</div>', unsafe_allow_html=True)

    jd_tab1, jd_tab2 = st.tabs(["Paste Text", "Upload File"])

    with jd_tab1:
        jd_text_input = st.text_area(
            "Paste the full job description here",
            height=280,
            placeholder="e.g.\n\nWe are looking for a Machine Learning Engineer with:\n\n• 3+ years Python experience\n• Strong skills in TensorFlow or PyTorch\n• Experience with MLOps, Docker, Kubernetes\n• NLP experience preferred\n• Bachelor's or Master's in CS or related field",
            label_visibility="collapsed",
        )
        job_title_input = st.text_input("Job Title (for reporting)", placeholder="e.g. Senior ML Engineer")

    with jd_tab2:
        jd_file = st.file_uploader("Upload JD (.txt, .pdf)", type=["txt", "pdf"])

with col_resume:
    st.markdown('<div class="section-header">👥 Resumes</div>', unsafe_allow_html=True)

    resume_tab1, resume_tab2 = st.tabs(["Upload Files", "Paste Texts"])

    with resume_tab1:
        resume_files = st.file_uploader(
            "Upload resumes (.txt, .pdf)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )

    with resume_tab2:
        n_paste = st.number_input("Number of candidates", min_value=1, max_value=10, value=2)
        paste_names = []
        paste_texts = []
        for i in range(int(n_paste)):
            with st.expander(f"Candidate {i+1}"):
                name = st.text_input(f"Name", key=f"name_{i}", value=f"Candidate {i+1}")
                text = st.text_area(f"Resume text", key=f"text_{i}", height=150,
                                     placeholder="Paste resume content here...")
                paste_names.append(name)
                paste_texts.append(text)


# ──────────────────────────────────────────────────────────────────────────────
# Demo Data Button
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("🎮 No data? Load demo resumes + JD"):
    st.markdown("Click to pre-fill with sample data and run a demo screening.")
    if st.button("Load Demo Data & Run"):
        st.session_state["demo"] = True
        st.rerun()



# ──────────────────────────────────────────────────────────────────────────────
# Run Button
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🚀 Screen Candidates", type="primary", use_container_width=True)

demo_mode = st.session_state.get("demo", False)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Execution
# ──────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------
# Cached Resources
# ---------------------------------------------------
@st.cache_resource
def get_scorer(cfg):
    from src.scorer import ResumeScorer
    return ResumeScorer(cfg)


@st.cache_resource
def get_pipeline():
    from src.pipeline import process_text
    return process_text


# ---------------------------------------------------
# Main Run Logic
# ---------------------------------------------------
if run_btn or demo_mode:

    if demo_mode:
        st.session_state["demo"] = False

        jd_raw = DEMO_JD
        job_title = "Senior Machine Learning Engineer"

        resume_inputs = [
            (r["name"], r["text"])
            for r in DEMO_RESUMES
        ]

    else:
        if jd_file:
            tmp = save_uploaded_file(jd_file)

            from src.preprocessor import extract_text_from_file

            jd_raw = extract_text_from_file(tmp)

        elif jd_text_input.strip():
            jd_raw = jd_text_input.strip()

        else:
            st.error(
                "Please provide a job description "
                "(paste text or upload file)."
            )
            st.stop()

        job_title = job_title_input or "the role"


        resume_inputs = []

        if resume_files:
            from src.preprocessor import extract_text_from_file

            for f in resume_files:
                tmp = save_uploaded_file(f)

                text = extract_text_from_file(tmp)

                resume_inputs.append(
                    (Path(f.name).stem, text)
                )

        else:
            for name, text in zip(paste_names, paste_texts):
                if text.strip():
                    resume_inputs.append((name, text))

        if not resume_inputs:
            st.error(
                "Please upload or paste at least one resume."
            )
            st.stop()


    if not config_valid:
        st.error(
            "Scoring weights must sum to 1.0. "
            "Adjust the sliders in the sidebar."
        )
        st.stop()

    cfg = ScoringConfig(
        semantic_weight=sem_w,
        skill_weight=sk_w,
        experience_weight=exp_w,
        semantic_backend=(
            "sentence_transformers"
            if use_st
            else "tfidf"
        ),
    )
    progress_bar = st.progress(0)
    status_text = st.empty()

    start = time.time()

    try:
        status_text.text("📦 Loading NLP pipeline...")
        progress_bar.progress(5)

        pt = get_pipeline()

        status_text.text("📄 Processing resumes...")
        progress_bar.progress(10)

        resume_data = []

        total_resumes = len(resume_inputs)

        for idx, (name, text) in enumerate(resume_inputs):

            data = pt(text)
            data["name"] = name

            resume_data.append(data)

            # Dynamic progress update
            progress = 10 + int(
                ((idx + 1) / total_resumes) * 40
            )

            progress_bar.progress(progress)

            status_text.text(
                f"📄 Processing resumes "
                f"({idx + 1}/{total_resumes})"
            )

        status_text.text("📝 Processing job description...")
        progress_bar.progress(55)

        jd_data = pt(jd_raw)

        status_text.text("🧠 Loading scoring model...")
        progress_bar.progress(65)

        scorer = get_scorer(cfg)


        status_text.text("🔍 Analyzing resumes...")
        progress_bar.progress(75)

        score_results = scorer.score_batch(
            resume_data,
            jd_data
        )


        status_text.text("📊 Ranking candidates...")
        progress_bar.progress(90)

        from src.ranker import (
            rank_candidates,
            summary_report,
        )

        ranked_df = rank_candidates(
            score_results,
            [r["name"] for r in resume_data],
            jd_data,
        )

        summary = summary_report(
            ranked_df,
            job_title,
        )

        elapsed = time.time() - start

        progress_bar.progress(100)

        status_text.text(
            "✅ Analysis complete."
        )

        st.success(
            f"✅ Screened "
            f"{summary['total_candidates']} "
            f"candidates in "
            f"{elapsed:.1f}s"
        )

    except Exception as e:
        st.error(f"Pipeline error: {e}")

        import traceback

        st.code(traceback.format_exc())

        st.stop()

    # ── Summary Metrics ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Screening Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, value, color in [
        (m1, "Candidates", summary["total_candidates"], "#f8fafc"),
        (m2, "Avg Score", summary["average_score"], "#fbbf24"),
        (m3, "Strong Match", summary["strong_matches"], "#22c55e"),
        (m4, "Good Match", summary["good_matches"], "#84cc16"),
        (m5, "Top Candidate", summary["top_candidate"][:15] + ("…" if len(summary.get("top_candidate","")) > 15 else ""), "#818cf8"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Visualizations ───────────────────────────────────────────────────────
    st.markdown("---")
    viz1, viz2 = st.columns([3, 2], gap="large")

    with viz1:
        st.markdown("**Candidate Comparison (Radar)**")
        st.plotly_chart(
            comparison_radar(ranked_df, top_n=min(5, len(ranked_df))),
            use_container_width=True
        )

    with viz2:
        st.markdown("**Match Tier Breakdown**")
        st.plotly_chart(tier_donut(summary), use_container_width=True)

        # Skill gap bar
        if summary.get("common_skill_gaps"):
            st.markdown("**Most Common Skill Gaps**")
            gaps_df = pd.DataFrame(summary["common_skill_gaps"], columns=["skill", "count"])
            fig_gaps = px.bar(
                gaps_df, x="count", y="skill", orientation="h",
                color="count",
                color_continuous_scale=["#1e293b", "#ef4444"],
            )
            fig_gaps.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(tickfont={"color": "#cbd5e1"}),
                xaxis=dict(tickfont={"color": "#64748b"}, showgrid=False),
            )
            st.plotly_chart(fig_gaps, use_container_width=True)

    # ── Ranked Candidates ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🏆 Ranked Candidates</div>', unsafe_allow_html=True)

    for rank, (_, row) in enumerate(ranked_df.iterrows(), 1):
        tier_colors_map = {
            "Strong Match": "#22c55e",
            "Good Match": "#84cc16",
            "Partial Match": "#f59e0b",
            "Weak Match": "#ef4444",
        }
        tc = tier_colors_map.get(row["tier"], "#64748b")

        with st.expander(
            f"#{rank}  {row['name']}  —  {row['final_score_pct']}  [{row['tier']}]",
            expanded=(rank <= 2),
        ):
            left, right = st.columns([2, 1], gap="large")

            with left:
                # Signal breakdown
                st.markdown("**Signal Breakdown**")
                signals = {
                    "Semantic Similarity": {
                        "score": row["semantic_score"],
                        "label": "Semantic Similarity",
                        "weight": cfg.semantic_weight,
                        "weighted": row["semantic_score"] * cfg.semantic_weight,
                    },
                    "Skill Match": {
                        "score": row["skill_match_score"],
                        "label": "Skill Match",
                        "weight": cfg.skill_weight,
                        "weighted": row["skill_match_score"] * cfg.skill_weight,
                    },
                    "Experience": {
                        "score": row["experience_score"],
                        "label": "Experience & Education",
                        "weight": cfg.experience_weight,
                        "weighted": row["experience_score"] * cfg.experience_weight,
                    },
                }
                st.plotly_chart(signal_bar_chart(signals), use_container_width=True)

                # Skills
                st.markdown("**Skills Analysis**")
                skill_html = ""
                for sk in sorted(row["matched_skills"])[:12]:
                    skill_html += f'<span class="skill-tag skill-matched">✓ {sk}</span>'
                for sk in sorted(row["missing_skills"])[:12]:
                    skill_html += f'<span class="skill-tag skill-missing">✗ {sk}</span>'
                for sk in sorted(row["bonus_skills"])[:6]:
                    skill_html += f'<span class="skill-tag skill-bonus">+ {sk}</span>'
                st.markdown(skill_html, unsafe_allow_html=True)
                st.caption("✓ Matched  ✗ Missing  + Bonus (not required)")

            with right:
                # Gauge
                st.plotly_chart(
                    score_gauge(row["final_score"], "Overall Score"),
                    use_container_width=True
                )

                # Profile info
                edu_labels = {
                    "phd": "PhD", "masters": "Master's", "bachelors": "Bachelor's",
                    "associate": "Associate", "diploma": "Diploma", "none": "Not specified"
                }
                st.markdown(f"""
                **Education:** {edu_labels.get(row['education_level'], 'N/A')}  
                **Experience:** {f"{row['years_experience']:.0f} yrs" if row['years_experience'] else 'Not specified'}  
                **Skills matched:** {row['matched_count']} / {row['total_required']}  
                **Skills missing:** {row['missing_count']}  
                **Bonus skills:** {row['bonus_count']}
                """)

                if row.get("email"):
                    st.caption(f"📧 {row['email']}")
                if row.get("phone"):
                    st.caption(f"📱 {row['phone']}")

            # Explanation
            import html
            st.markdown("---")

            explanation_text = str(row.get("explanation", "")).strip()

            if explanation_text.startswith("DeltaGenerator"):
                explanation_text = ""

            if explanation_text:
                st.markdown("**💡 Why this score?**")

                explanation_text = html.escape(explanation_text)

                for line in explanation_text.split("\n\n"):
                    line = line.strip()
                    if line:
                        st.markdown(line)
    

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)

    export_df = ranked_df.copy()
    for col in ["matched_skills", "missing_skills", "bonus_skills"]:
        export_df[col] = export_df[col].apply(lambda x: " | ".join(x) if isinstance(x, list) else x)
    if "explanation" in export_df.columns:
        export_df = export_df.drop(columns=["explanation"])
    if "tier_color" in export_df.columns:
        export_df = export_df.drop(columns=["tier_color"])

    csv_bytes = export_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        "⬇️ Download Rankings CSV",
        data=csv_bytes,
        file_name=f"resumeiq_results_{job_title.replace(' ', '_')}.csv",
        mime="text/csv",
    )

else:
    # Welcome state
    st.markdown("""
    <div class="info-box">
        <strong>How to use ResumeIQ:</strong><br><br>
        1. Paste or upload a <strong>job description</strong> on the left<br>
        2. Upload <strong>resume files</strong> (PDF/TXT) or paste resume text on the right<br>
        3. Optionally adjust <strong>scoring weights</strong> in the sidebar<br>
        4. Click <strong>Screen Candidates</strong> to run the ML pipeline<br><br>
        👆 Try the <strong>Load Demo Data</strong> section above to see a live example instantly.
    </div>
    """, unsafe_allow_html=True)
