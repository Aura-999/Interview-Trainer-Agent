"""
resume_parser.py — Extract structured candidate profile from PDF/DOCX resume.
Keeps it lightweight: regex + keyword matching (no spaCy model download needed).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


# ── Skill taxonomy (expand as needed) ────────────────────────────────────────

SKILL_KEYWORDS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "scala", "r", "matlab", "bash", "shell",
    # Web
    "react", "angular", "vue", "next.js", "node.js", "html", "css",
    "fastapi", "django", "flask", "spring", "express",
    # Data / ML
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "spark", "hadoop", "kafka", "airflow", "dbt",
    # Cloud / DevOps
    "aws", "azure", "gcp", "ibm cloud", "docker", "kubernetes", "terraform",
    "git", "github", "gitlab", "ci/cd", "linux",
    # Other
    "machine learning", "deep learning", "nlp", "computer vision",
    "data analysis", "data visualization", "restful api", "graphql",
    "agile", "scrum",
}

EXPERIENCE_PATTERNS = [
    # "5 years", "5+ years", "5-7 years"
    r"(\d+)\+?\s*(?:to|-)\s*\d*\s*years?\s+(?:of\s+)?(?:experience|exp)",
    r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)",
    # "experience: 3 years"
    r"(?:experience|exp)[:\s]+(\d+)\+?\s*years?",
]

SECTION_HEADERS = re.compile(
    r"^\s*(experience|work experience|education|skills|projects|"
    r"certifications|summary|objective|profile)\s*$",
    re.IGNORECASE,
)


@dataclass
class CandidateProfile:
    raw_text: str = ""
    skills: list[str] = field(default_factory=list)
    years_experience: int = 0
    experience_level: str = "junior"   # junior / mid / senior
    education: str = ""
    current_role: str = ""
    projects_summary: str = ""


def _extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _extract_text_from_docx(path: Path) -> str:
    from docx import Document  # lazy import
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        # word-boundary aware match
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def _extract_years_experience(text: str) -> int:
    text_lower = text.lower()
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return min(int(match.group(1)), 30)  # sanity cap
    # Fallback: count unique years in date ranges (e.g. "2019 – 2023")
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
    if len(years) >= 2:
        years_int = sorted(set(int(y) for y in years))
        span = years_int[-1] - years_int[0]
        return min(span, 20)
    return 0


def _years_to_level(years: int) -> str:
    if years <= 2:
        return "junior"
    if years <= 5:
        return "mid"
    return "senior"


def _extract_education(text: str) -> str:
    degrees = re.findall(
        r"(b\.?tech|b\.?e\.?|b\.?sc|m\.?tech|m\.?sc|m\.?s\.?|phd|bachelor|master|mba)[^.\n]{0,60}",
        text,
        re.IGNORECASE,
    )
    return degrees[0].strip() if degrees else ""


def _extract_current_role(text: str) -> str:
    """Grab the first job title-like line from the top of the resume."""
    title_patterns = [
        r"(software engineer|data scientist|frontend developer|backend developer|"
        r"full[ -]stack developer|machine learning engineer|devops engineer|"
        r"product manager|data analyst|solutions architect)[^\n]{0,40}",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, text[:1500], re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def _extract_projects_summary(text: str) -> str:
    """Return first 300 chars under a Projects section, if present."""
    match = re.search(
        r"projects?\s*\n(.*?)(?:\n[A-Z][A-Z\s]{3,}\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()[:300]
    return ""


def parse_resume(file_path: str | Path) -> CandidateProfile:
    """
    Parse a resume file (PDF or DOCX) and return a CandidateProfile.
    Falls back gracefully — partial data is better than an exception.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = _extract_text_from_pdf(path)
    elif suffix in {".docx", ".doc"}:
        raw_text = _extract_text_from_docx(path)
    else:
        # Treat as plain text
        raw_text = path.read_text(encoding="utf-8", errors="ignore")

    years = _extract_years_experience(raw_text)
    profile = CandidateProfile(
        raw_text=raw_text,
        skills=_extract_skills(raw_text),
        years_experience=years,
        experience_level=_years_to_level(years),
        education=_extract_education(raw_text),
        current_role=_extract_current_role(raw_text),
        projects_summary=_extract_projects_summary(raw_text),
    )
    return profile


def parse_resume_from_bytes(file_bytes: bytes, filename: str) -> CandidateProfile:
    """Parse from in-memory bytes (used by FastAPI upload handler)."""
    import tempfile
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    profile = parse_resume(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return profile
