# 🎯 ResumeIQ

[![CI](https://github.com/anaboset/FUTURE_ML_03/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/resumeiq/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP.streamlit.app)

> **Automatically screen, score, and rank job candidates using a 3-signal ML ensemble. Built for HR teams, recruiters, and HR-tech platforms.**


---

## 🖥️[Live Demo](https://streamlit.app.io)

👉 **[Try the live app →](https://YOUR_APP.streamlit.app)**

![ResumeIQ Dashboard](data/processed/demo_screenshot.png)

---

## 🧠 How It Works

ResumeIQ uses a **3-signal ensemble scoring model** — the same architectural pattern used in real HR-tech products.

```
Resume PDF/Text ──┐
                  ├──► Preprocessor ──► Skill Extractor ──► Scorer ──► Ranker ──► Dashboard
Job Description ──┘
```

### Scoring Architecture

| Signal | Weight | Method | What it captures |
|--------|--------|--------|-----------------|
| 🔵 Semantic Similarity | **45%** | `sentence-transformers` (MiniLM-L6-v2) cosine similarity | Overall domain & language alignment |
| 🟢 Skill Match | **40%** | Taxonomy-based exact + alias matching (200+ skills) | Required skill coverage & gaps |
| 🟣 Experience Score | **15%** | Education level + years of experience heuristics | Seniority & qualification fit |

**Weights are fully configurable** via the dashboard sidebar.

### Why sentence-transformers over TF-IDF?

TF-IDF treats "built neural networks" and "deep learning experience" as completely different. Sentence-transformers understand semantic equivalence — a resume that says "I built transformer models in PyTorch" will correctly match a JD asking for "deep learning with PyTorch experience."

---

## ✨ Features

- **📄 Multi-format input** — Upload PDF or TXT resumes, or paste text directly
- **🔍 NLP skill extraction** — 200+ skill taxonomy covering ML, cloud, data, web, soft skills
- **📊 Multi-signal scoring** — 3 independent signals combined via weighted ensemble
- **🏆 Candidate ranking** — Sorted results with tier classification (Strong/Good/Partial/Weak match)
- **💡 Explainability** — Plain-English explanation for every candidate's score
- **❌ Skill gap analysis** — Exactly which skills are matched, missing, and extra
- **📈 Visual dashboard** — Radar chart, gauge, bar charts, donut breakdown
- **⚙️ Configurable weights** — Adjust signal weights in real time
- **📥 CSV export** — Download full ranked results

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/resumeiq.git
cd resumeiq

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the dashboard

```bash
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) — click **Load Demo Data** to see it in action immediately.

### 3. Use as a library

```python
from src.pipeline import ResumeIQPipeline

pipeline = ResumeIQPipeline()

ranked_df, summary = pipeline.run(
    resumes=["path/to/resume1.pdf", "path/to/resume2.pdf"],
    job_description="path/to/jd.txt",
    job_title="Senior ML Engineer",
)

print(ranked_df[["name", "final_score_pct", "tier", "matched_skills", "missing_skills"]])
```

---

## 📁 Project Structure

```
resumeiq/
│
├── src/                        # Core ML pipeline (production-ready modules)
│   ├── __init__.py
│   ├── preprocessor.py         # Text cleaning, PDF extraction, lemmatization
│   ├── skill_extractor.py      # 200+ skill taxonomy + NER + contact extraction
│   ├── scorer.py               # 3-signal ensemble scoring engine
│   ├── ranker.py               # Ranking, explainability, summary reports
│   └── pipeline.py             # Orchestration layer
│
├── app/
│   └── dashboard.py            # Streamlit interactive dashboard
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_scoring_experiments.ipynb  # Model validation & experiments
│
├── tests/
│   └── test_scorer.py          # 30+ unit + integration tests
│
├── data/
│   ├── raw/                    # Place Kaggle dataset here (gitignored)
│   └── processed/              # Cleaned data & visualizations
│
├── .github/workflows/ci.yml    # GitHub Actions CI (Python 3.10 & 3.11)
├── .streamlit/config.toml      # Streamlit theme config
├── requirements.txt
└── README.md
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite covers:
- Text preprocessing edge cases
- Skill extraction accuracy (aliases, regex patterns)
- Scoring signal correctness (perfect match, zero match, partial)
- Ranking order consistency
- Full pipeline integration test

---

## 📊 Dataset

This project is compatible with the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).

Download and place at: `data/raw/UpdatedResumeDataSet.csv`

The EDA notebook (`notebooks/01_eda.ipynb`) handles the rest.

---

## 🔬 Design Decisions

**Why not a trained classification model?**  
A trained classifier requires labeled pairs of (resume, JD) with binary match labels — data that's expensive to obtain and domain-specific. The ensemble approach generalizes across any job role without retraining, and provides interpretable output (critical for HR compliance).

**Why sentence-transformers over TF-IDF?**  
TF-IDF is sensitive to vocabulary mismatch. Semantic embeddings capture meaning — "built production ML pipelines" and "MLOps engineering experience" score high similarity even with no word overlap.

**Why three signals?**  
Each signal captures a different dimension: semantic captures overall domain fit, skill match captures concrete technical alignment, and experience captures career stage fit. Combining them makes the score more robust than any single signal.

**Why configurable weights?**  
Different roles have different priorities. A research role may weight semantic similarity higher; a junior role may care less about experience. The sidebar lets recruiters tune for each position.

---

## 🗺️ Potential Extensions

- [ ] Fine-tuned domain-specific embedding model on HR data
- [ ] Named entity recognition for job titles and companies
- [ ] Bias detection / fairness audit module
- [ ] REST API with FastAPI for production integration
- [ ] Bulk processing from Google Drive / SharePoint
- [ ] ATS (Applicant Tracking System) export format

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| NLP | spaCy, NLTK, sentence-transformers |
| ML | scikit-learn, numpy |
| PDF parsing | pdfplumber |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, pytest-cov |
| CI/CD | GitHub Actions |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built during the Future Interns ML Internship (2026).

---
