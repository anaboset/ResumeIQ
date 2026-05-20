
# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(251, 191, 36, 0.15);
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #f8fafc;
    margin: 0;
    line-height: 1.1;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

.accent { color: #fbbf24; }

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #f8fafc;
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

.candidate-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.candidate-card:hover {
    border-color: #fbbf24;
}

.rank-badge {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #475569;
    border-radius: 50%;
    width: 2rem;
    height: 2rem;
    line-height: 2rem;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
    color: #fbbf24;
    margin-right: 0.75rem;
}

.tier-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.skill-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    margin: 0.15rem;
    font-weight: 500;
}

.skill-matched { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.skill-missing  { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.skill-bonus    { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f8fafc;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.info-box {
    background: rgba(251, 191, 36, 0.05);
    border: 1px solid rgba(251, 191, 36, 0.2);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.9rem;
    color: #cbd5e1;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""