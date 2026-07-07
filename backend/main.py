"""
main.py — FastAPI application.

Endpoints:
  POST /api/parse-resume        → CandidateProfile (JSON)
  POST /api/generate-questions  → list of Question objects
  POST /api/generate-answer     → ModelAnswer
  POST /api/get-feedback        → FeedbackResult
  GET  /api/health              → status check
"""

import sys
import os

# Ensure backend package is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import logging
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from prompt_builder import (
    build_feedback_prompt,
    build_model_answer_prompt,
    build_question_gen_prompt,
)
from rag_engine import retrieve_for_answer, retrieve_for_question_gen
from resume_parser import CandidateProfile, parse_resume_from_bytes
from watsonx_client import generate

load_dotenv()

_log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: warn if IBM credentials are still placeholders."""
    api_key = os.getenv("WATSONX_API_KEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    if api_key in {"your_watsonx_api_key_here", ""} or project_id in {"your_watsonx_project_id_here", ""}:
        _log.warning(
            "\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║  ⚠  IBM watsonx credentials not configured           ║\n"
            "║                                                      ║\n"
            "║  Set in .env:                                        ║\n"
            "║    WATSONX_API_KEY=<your real key>                   ║\n"
            "║    WATSONX_PROJECT_ID=<your real project id>         ║\n"
            "║                                                      ║\n"
            "║  Get them at: https://cloud.ibm.com                  ║\n"
            "║  Resume parsing + RAG retrieval still work fine.     ║\n"
            "╚══════════════════════════════════════════════════════╝"
        )
    yield   # app runs here


app = FastAPI(title="Interview Trainer API", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ParsedProfile(BaseModel):
    skills: list[str]
    years_experience: int
    experience_level: str
    education: str
    current_role: str
    projects_summary: str


class Question(BaseModel):
    number: int
    text: str
    type: Literal["Technical", "Behavioral"]
    difficulty: Literal["Easy", "Medium", "Hard"]
    why: str


class GenerateQuestionsRequest(BaseModel):
    target_role: str
    skills: list[str]
    years_exp: int
    experience_level: str
    education: str
    projects_summary: str
    num_questions: int = 8


class GenerateAnswerRequest(BaseModel):
    question: str
    question_type: Literal["Technical", "Behavioral"]
    target_role: str
    experience_level: str
    skills: list[str]


class GetFeedbackRequest(BaseModel):
    question: str
    question_type: Literal["Technical", "Behavioral"]
    user_answer: str
    model_answer: str


class ModelAnswer(BaseModel):
    answer: str
    cached: bool = False


class FeedbackResult(BaseModel):
    raw: str
    scores: dict[str, int]
    overall: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_questions(raw: str) -> list[Question]:
    """
    Parse Granite output into structured Question objects.
    Format expected per question:
        Q1. [Technical|Behavioral] [Easy|Medium|Hard]
        <question text>
        Why: <reason>
    """
    import re
    questions = []
    blocks = re.split(r"\nQ\d+\.", "\nQ1." + raw.split("Q1.")[-1] if "Q1." in raw else raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        # First line: header tags
        header = lines[0]
        q_type: Literal["Technical", "Behavioral"] = "Technical"
        difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"

        if "Behavioral" in header:
            q_type = "Behavioral"
        for d in ("Easy", "Medium", "Hard"):
            if d in header:
                difficulty = d
                break

        # Second line onwards: question text until "Why:"
        question_lines = []
        why = ""
        for line in lines[1:]:
            if line.lower().startswith("why:"):
                why = line[4:].strip()
            else:
                question_lines.append(line)

        q_text = " ".join(question_lines).strip()
        if not q_text:
            continue

        questions.append(
            Question(
                number=len(questions) + 1,
                text=q_text,
                type=q_type,
                difficulty=difficulty,
                why=why,
            )
        )
    return questions


def _parse_feedback_scores(raw: str) -> tuple[dict[str, int], int]:
    """Extract score dictionary and overall total from feedback text."""
    import re
    criteria = ["Relevance", "Depth", "Structure", "Clarity", "Completeness"]
    scores: dict[str, int] = {}
    for c in criteria:
        m = re.search(rf"{c}:\s*(\d)/5", raw)
        scores[c] = int(m.group(1)) if m else 0

    m_overall = re.search(r"Overall:\s*(\d+)/25", raw)
    overall = int(m_overall.group(1)) if m_overall else sum(scores.values())
    return scores, overall


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/parse-resume", response_model=ParsedProfile)
async def parse_resume_endpoint(file: UploadFile = File(...)):
    if file.content_type not in {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, DOCX, or TXT.")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5 MB cap
        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")

    profile: CandidateProfile = parse_resume_from_bytes(contents, file.filename or "resume.pdf")
    return ParsedProfile(
        skills=profile.skills,
        years_experience=profile.years_experience,
        experience_level=profile.experience_level,
        education=profile.education,
        current_role=profile.current_role,
        projects_summary=profile.projects_summary,
    )


@app.post("/api/generate-questions", response_model=list[Question])
async def generate_questions(req: GenerateQuestionsRequest):
    tech_chunks, beh_chunks = retrieve_for_question_gen(
        target_role=req.target_role,
        skills=req.skills,
        experience_level=req.experience_level,
        n_technical=4,
        n_behavioral=3,
    )

    prompt = build_question_gen_prompt(
        target_role=req.target_role,
        skills=req.skills,
        years_exp=req.years_exp,
        experience_level=req.experience_level,
        education=req.education,
        projects_summary=req.projects_summary,
        tech_chunks=tech_chunks,
        beh_chunks=beh_chunks,
        num_questions=req.num_questions,
    )

    # Cache key: role + level + top 4 skills (stable enough across same profile)
    cache_parts = [req.target_role, req.experience_level, ",".join(sorted(req.skills[:4]))]
    raw = generate(prompt, task="questions", cache_key_parts=cache_parts)

    questions = _parse_questions(raw)
    if not questions:
        raise HTTPException(status_code=502, detail="Could not parse questions from model output.")
    return questions


@app.post("/api/generate-answer", response_model=ModelAnswer)
async def generate_answer(req: GenerateAnswerRequest):
    from rag_engine import get_cached_response, make_cache_key, set_cached_response

    # Tight cache key — same question + role + level = same model answer
    cache_parts = [req.question, req.target_role, req.experience_level, req.question_type]
    cache_key = make_cache_key("answer", *cache_parts)
    cached = get_cached_response(cache_key)
    if cached:
        return ModelAnswer(answer=cached, cached=True)

    answer_chunks = retrieve_for_answer(
        question=req.question,
        question_type=req.question_type,
        role=req.target_role,
    )
    prompt = build_model_answer_prompt(
        question=req.question,
        question_type=req.question_type,
        experience_level=req.experience_level,
        target_role=req.target_role,
        skills_context=", ".join(req.skills[:6]),
        answer_chunks=answer_chunks,
    )
    answer = generate(prompt, task="answer", cache_key_parts=cache_parts)
    set_cached_response(cache_key, answer)
    return ModelAnswer(answer=answer, cached=False)


@app.post("/api/get-feedback", response_model=FeedbackResult)
async def get_feedback(req: GetFeedbackRequest):
    prompt = build_feedback_prompt(
        question=req.question,
        question_type=req.question_type,
        user_answer=req.user_answer,
        model_answer=req.model_answer,
    )
    # Feedback is always fresh — user answers differ every time
    raw = generate(prompt, task="feedback")
    scores, overall = _parse_feedback_scores(raw)
    return FeedbackResult(raw=raw, scores=scores, overall=overall)
