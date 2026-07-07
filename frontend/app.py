"""
app.py — Streamlit frontend for the Interview Trainer Agent.

Run:
    streamlit run frontend/app.py
"""

import json
import sys
import os
from pathlib import Path

import httpx
import plotly.graph_objects as go
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")
ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Frontend Engineer",
    "Backend Engineer",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "Data Analyst",
]

st.set_page_config(
    page_title="Interview Trainer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }

    /* ── Sidebar shell ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Section labels (uppercase headings we inject) */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e2e8f0;
    }

    /* Native widget labels */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio   label,
    [data-testid="stSidebar"] .stSlider  label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #94a3b8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Selectbox — white bg, dark text ── */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    /* Dropdown chevron + placeholder */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg { fill: #475569 !important; }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }

    /* ── Text input — white bg, dark text ── */
    [data-testid="stSidebar"] .stTextInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }

    /* ── File uploader ── */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px dashed #475569 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }

    /* ── Radio buttons — light labels ── */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
        color: #e2e8f0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-transform: capitalize !important;
        letter-spacing: 0 !important;
    }

    /* ── Slider — track + thumb ── */
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stSlider p {
        color: #e2e8f0 !important;
    }

    /* ── Generate button ── */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 12px 0 !important;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 14px rgba(99,102,241,0.45) !important;
        transition: opacity .2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* ── Main header ── */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        color: #fff;
    }
    .hero h1 { margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; }
    .hero p  { margin: 6px 0 0; font-size: 15px; opacity: 0.88; }

    /* ── Stats bar ── */
    .stats-bar {
        display: flex;
        gap: 14px;
        margin-bottom: 22px;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 13px;
        font-weight: 600;
        color: #475569;
    }
    .stat-chip span { color: #6366f1; }

    /* ── Question card ── */
    .q-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .q-card:hover { border-left-color: #8b5cf6; }
    .q-number {
        font-size: 11px;
        font-weight: 700;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .q-text {
        font-size: 16px;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .q-why {
        font-size: 12px;
        color: #64748b;
        font-style: italic;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-right: 6px;
        text-transform: uppercase;
    }
    .badge-technical  { background:#e0e7ff; color:#4338ca; }
    .badge-behavioral { background:#f3e8ff; color:#7c3aed; }
    .badge-easy   { background:#dcfce7; color:#15803d; }
    .badge-medium { background:#fef9c3; color:#a16207; }
    .badge-hard   { background:#fee2e2; color:#b91c1c; }

    /* ── Score display ── */
    .score-ring {
        text-align: center;
        padding: 18px;
        background: linear-gradient(135deg, #f0f9ff, #e0e7ff);
        border-radius: 14px;
        margin-bottom: 12px;
    }
    .score-number { font-size: 42px; font-weight: 800; color: #6366f1; line-height: 1; }
    .score-label  { font-size: 12px; color: #64748b; margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }

    /* ── Score rows ── */
    .score-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 13px;
    }
    .score-row:last-child { border-bottom: none; }
    .score-row .label { color: #475569; font-weight: 500; }
    .score-row .val   { font-weight: 700; }

    /* ── Feedback box ── */
    .feedback-prose {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 14px;
        color: #1e293b;
        line-height: 1.7;
        border: 1px solid #e2e8f0;
    }

    /* ── Profile card in sidebar ── */
    .profile-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 8px;
    }
    .profile-card .row { font-size: 12px; color: #cbd5e1; margin-bottom: 4px; }
    .profile-card .row b { color: #e2e8f0; }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #94a3b8;
    }
    .empty-state .icon { font-size: 52px; margin-bottom: 14px; }
    .empty-state h3 { color: #475569; font-size: 20px; margin-bottom: 8px; }
    .empty-state p  { font-size: 14px; max-width: 380px; margin: 0 auto; }

    /* ── Divider override ── */
    hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "profile": None,
        "questions": [],
        "model_answers": {},
        "feedback": {},
        "active_q": None,
        "target_role": ROLES[0],
        "manual_skills": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

def api_post(endpoint: str, **kwargs) -> dict | list:
    try:
        r = httpx.post(f"{API_BASE}/{endpoint}", timeout=60.0, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        st.stop()
    except httpx.ConnectError:
        st.error("Cannot connect to backend. Is `uvicorn backend.main:app` running?")
        st.stop()


# ── Sidebar — Input ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
        <div style="font-size:24px; font-weight:800; color:#e2e8f0; letter-spacing:-0.5px;">🎯 Interview Trainer</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">Powered by IBM watsonx.ai + RAG</div>
    </div>
    """, unsafe_allow_html=True)

    target_role = st.selectbox("Target Role", ROLES, key="target_role")
    exp_level   = st.radio("Experience Level", ["junior", "mid", "senior"], horizontal=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Upload Resume</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    st.markdown('<p style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;margin-top:8px;">Or Enter Skills</p>', unsafe_allow_html=True)
    manual_skills_input = st.text_input(
        "Skills",
        placeholder="Python, SQL, React…",
        key="manual_skills",
        label_visibility="collapsed",
    )

    num_q = st.slider("Questions", min_value=4, max_value=12, value=8, step=2)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    generate_btn = st.button("🚀  Generate Interview", type="primary", use_container_width=True)

    if st.session_state.profile:
        p = st.session_state.profile
        skills_preview = ", ".join(p["skills"][:6]) + ("…" if len(p["skills"]) > 6 else "")
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:11px;font-weight:700;color:#22d3ee;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">✓ Resume Parsed</div>
            <div class="row"><b>Role:</b> {p.get("current_role") or "N/A"}</div>
            <div class="row"><b>Experience:</b> {p["years_experience"]} yrs ({p["experience_level"]})</div>
            <div class="row"><b>Skills:</b> {skills_preview}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main Area ─────────────────────────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero">
    <h1>🎯 Interview Trainer Agent</h1>
    <p>AI-powered personalised interview prep with model answers, scoring, and feedback</p>
</div>
""", unsafe_allow_html=True)

# ── Step 1: Parse resume on upload ───────────────────────────────────────────

if uploaded and st.session_state.profile is None:
    with st.spinner("Parsing resume…"):
        result = api_post(
            "parse-resume",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
        )
        st.session_state.profile = result
        st.rerun()

# ── Step 2: Generate questions ───────────────────────────────────────────────

if generate_btn:
    profile = st.session_state.profile or {}
    skills = list(profile.get("skills", []))
    if st.session_state.manual_skills.strip():
        extra = [s.strip().lower() for s in st.session_state.manual_skills.split(",") if s.strip()]
        skills = list(dict.fromkeys(skills + extra))

    if not skills:
        st.warning("Add at least one skill or upload a resume so we can personalise the questions.")
        st.stop()

    with st.spinner("Retrieving knowledge base + generating questions via IBM watsonx…"):
        questions = api_post(
            "generate-questions",
            json={
                "target_role":      target_role,
                "skills":           skills[:12],
                "years_exp":        profile.get("years_experience", 0),
                "experience_level": exp_level,
                "education":        profile.get("education", ""),
                "projects_summary": profile.get("projects_summary", ""),
                "num_questions":    num_q,
            },
        )
        st.session_state.questions     = questions
        st.session_state.model_answers = {}
        st.session_state.feedback      = {}
        st.session_state.active_q      = None

# ── Step 3: Display questions ────────────────────────────────────────────────

questions = st.session_state.questions
if not questions:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💼</div>
        <h3>Ready when you are</h3>
        <p>Upload your resume or enter your skills in the sidebar, then click <strong>Generate Interview</strong> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Stats bar
tech_count = sum(1 for q in questions if q["type"] == "Technical")
beh_count  = sum(1 for q in questions if q["type"] == "Behavioral")
answered   = len(st.session_state.feedback)

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-chip">🎯 Role: <span>{target_role}</span></div>
    <div class="stat-chip">📋 <span>{len(questions)}</span> Questions</div>
    <div class="stat-chip">⚙️ <span>{tech_count}</span> Technical</div>
    <div class="stat-chip">🧠 <span>{beh_count}</span> Behavioral</div>
    <div class="stat-chip">✅ <span>{answered}</span> Answered</div>
</div>
""", unsafe_allow_html=True)

# ── Question cards ────────────────────────────────────────────────────────────

for i, q in enumerate(questions):
    type_badge = (
        '<span class="badge badge-technical">Technical</span>'
        if q["type"] == "Technical"
        else '<span class="badge badge-behavioral">Behavioral</span>'
    )
    diff_badge = f'<span class="badge badge-{q["difficulty"].lower()}">{q["difficulty"]}</span>'

    st.markdown(
        f'<div class="q-card">'
        f'<div class="q-number">Question {q["number"]} of {len(questions)}</div>'
        f'{type_badge}{diff_badge}'
        f'<div class="q-text">{q["text"]}</div>'
        f'<div class="q-why">💡 {q["why"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💡 Show Model Answer", key=f"ans_btn_{i}", use_container_width=True):
            if i not in st.session_state.model_answers:
                with st.spinner("Generating model answer…"):
                    profile = st.session_state.profile or {}
                    skills  = list(profile.get("skills", []))
                    if st.session_state.manual_skills.strip():
                        skills += [s.strip().lower() for s in st.session_state.manual_skills.split(",")]
                    result = api_post(
                        "generate-answer",
                        json={
                            "question":         q["text"],
                            "question_type":    q["type"],
                            "target_role":      target_role,
                            "experience_level": exp_level,
                            "skills":           skills[:8],
                        },
                    )
                    st.session_state.model_answers[i] = result["answer"]
                    if result.get("cached"):
                        st.caption("⚡ Served from cache — 0 tokens used")
            st.session_state.active_q = i

    if i in st.session_state.model_answers:
        with st.expander("📖 Model Answer", expanded=(st.session_state.active_q == i)):
            st.markdown(
                f'<div class="feedback-prose">{st.session_state.model_answers[i]}</div>',
                unsafe_allow_html=True,
            )

    # ── Step 4: User answer + feedback ───────────────────────────────────────
    user_ans_key = f"user_ans_{i}"
    user_answer  = st.text_area(
        f"✍️ Your Answer",
        key=user_ans_key,
        height=120,
        placeholder="Type your answer here, then click Get Feedback…",
    )

    with col2:
        if st.button("📊 Get Feedback", key=f"fb_btn_{i}", use_container_width=True):
            if not user_answer.strip():
                st.warning("Write your answer first.")
            else:
                model_ans = st.session_state.model_answers.get(i)
                if model_ans is None:
                    with st.spinner("Fetching model answer for comparison…"):
                        profile = st.session_state.profile or {}
                        skills  = list(profile.get("skills", []))
                        res = api_post(
                            "generate-answer",
                            json={
                                "question":         q["text"],
                                "question_type":    q["type"],
                                "target_role":      target_role,
                                "experience_level": exp_level,
                                "skills":           skills[:8],
                            },
                        )
                        model_ans = res["answer"]
                        st.session_state.model_answers[i] = model_ans

                with st.spinner("Analysing your answer…"):
                    fb = api_post(
                        "get-feedback",
                        json={
                            "question":      q["text"],
                            "question_type": q["type"],
                            "user_answer":   user_answer,
                            "model_answer":  model_ans,
                        },
                    )
                    st.session_state.feedback[i] = fb

    if i in st.session_state.feedback:
        fb      = st.session_state.feedback[i]
        scores  = fb.get("scores", {})
        overall = fb.get("overall", 0)

        with st.expander("📊 Feedback & Scores", expanded=True):
            if criteria := list(scores.keys()):
                values = list(scores.values())
                if any(v > 0 for v in values):
                    # Radar chart
                    fig = go.Figure(
                        go.Scatterpolar(
                            r=values + [values[0]],
                            theta=criteria + [criteria[0]],
                            fill="toself",
                            fillcolor="rgba(99,102,241,0.18)",
                            line=dict(color="#6366f1", width=2.5),
                            marker=dict(color="#8b5cf6", size=6),
                            name="Your Score",
                        )
                    )
                    fig.update_layout(
                        polar=dict(
                            bgcolor="#f8fafc",
                            radialaxis=dict(
                                visible=True,
                                range=[0, 5],
                                tickfont_size=10,
                                gridcolor="#e2e8f0",
                            ),
                            angularaxis=dict(gridcolor="#e2e8f0"),
                        ),
                        showlegend=False,
                        margin=dict(t=30, b=20, l=40, r=40),
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        # Overall score ring
                        pct = int((overall / 25) * 100)
                        color = "#22c55e" if pct >= 72 else "#f59e0b" if pct >= 48 else "#ef4444"
                        st.markdown(f"""
                        <div class="score-ring">
                            <div class="score-number" style="color:{color}">{overall}</div>
                            <div class="score-label">out of 25</div>
                        </div>
                        """, unsafe_allow_html=True)
                        for k, v in scores.items():
                            dot = "🟢" if v >= 4 else "🟡" if v == 3 else "🔴"
                            st.markdown(
                                f'<div class="score-row"><span class="label">{k}</span>'
                                f'<span class="val">{dot} {v}/5</span></div>',
                                unsafe_allow_html=True,
                            )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="feedback-prose">{fb.get("raw", "")}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:32px"/>
<div style="text-align:center; padding:16px 0; font-size:12px; color:#94a3b8;">
    Interview Trainer Agent &nbsp;·&nbsp; IBM watsonx.ai &nbsp;·&nbsp; RAG with ChromaDB
</div>
""", unsafe_allow_html=True)
