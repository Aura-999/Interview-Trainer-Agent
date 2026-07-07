"""
prompt_builder.py — Loads prompt templates and fills them with runtime data.

Token-minimisation strategy:
  - Templates are dense and imperative — no pleasantries, no repeating the question back.
  - Retrieved context is hard-capped via format_chunks_for_prompt().
  - Skills list is capped at 6 items.
  - Project summary is capped at 120 chars.
"""

from pathlib import Path

from rag_engine import RetrievedChunk, format_chunks_for_prompt

PROMPTS_DIR = Path("./prompts")


def _load(template_name: str) -> str:
    return (PROMPTS_DIR / template_name).read_text(encoding="utf-8")


def build_question_gen_prompt(
    target_role: str,
    skills: list[str],
    years_exp: int,
    experience_level: str,
    education: str,
    projects_summary: str,
    tech_chunks: list[RetrievedChunk],
    beh_chunks: list[RetrievedChunk],
    num_questions: int = 8,
) -> str:
    template = _load("question_gen.txt")
    # Cap context size to control tokens
    tech_context = format_chunks_for_prompt(tech_chunks, max_chars=700)
    beh_context = format_chunks_for_prompt(beh_chunks, max_chars=500)
    combined_context = f"TECHNICAL REFERENCES:\n{tech_context}\n\nBEHAVIORAL REFERENCES:\n{beh_context}"

    return template.format(
        target_role=target_role,
        years_exp=years_exp,
        experience_level=experience_level,
        skills_list=", ".join(skills[:6]),                   # cap at 6 skills
        education=education or "Not specified",
        projects_summary=(projects_summary or "Not specified")[:120],  # cap chars
        retrieved_context=combined_context,
        num_questions=num_questions,
    )


def build_model_answer_prompt(
    question: str,
    question_type: str,
    experience_level: str,
    target_role: str,
    skills_context: str,
    answer_chunks: list[RetrievedChunk],
) -> str:
    template = _load("model_answer.txt")
    ref_context = format_chunks_for_prompt(answer_chunks, max_chars=600)
    return template.format(
        question=question,
        question_type=question_type,
        experience_level=experience_level,
        target_role=target_role,
        skills_context=skills_context[:200],  # cap
        retrieved_context=ref_context,
    )


def build_feedback_prompt(
    question: str,
    question_type: str,
    user_answer: str,
    model_answer: str,
) -> str:
    template = _load("feedback.txt")
    return template.format(
        question=question,
        question_type=question_type,
        # Cap user answer and model answer to limit tokens going into feedback
        user_answer=user_answer[:600],
        model_answer=model_answer[:500],
    )
