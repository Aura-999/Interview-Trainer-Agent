"""
rag_engine.py — Retrieval layer: query ChromaDB and return relevant chunks.
Includes a disk-based response cache to avoid re-calling Granite for identical inputs.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import diskcache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Singletons (loaded once per process) ────────────────────────────────────

_embedder: SentenceTransformer | None = None
_chroma_collection: chromadb.Collection | None = None
_response_cache: diskcache.Cache | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_or_create_collection(
            name="interview_kb",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def _get_cache() -> diskcache.Cache:
    global _response_cache
    if _response_cache is None:
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _response_cache = diskcache.Cache(CACHE_DIR)
    return _response_cache


# ── Cache helpers ────────────────────────────────────────────────────────────

def make_cache_key(*parts: str) -> str:
    """Deterministic cache key from arbitrary string parts."""
    combined = "||".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


def get_cached_response(key: str) -> str | None:
    return _get_cache().get(key)


def set_cached_response(key: str, value: str, expire: int = 86400 * 7) -> None:
    """Cache for 7 days by default — interview Q&A doesn't change often."""
    _get_cache().set(key, value, expire=expire)


# ── Retrieval ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    doc_type: str       # technical | behavioral | hr_guideline
    metadata: dict
    score: float        # cosine distance (lower = more similar)


def retrieve(
    query: str,
    doc_type: str | None = None,   # filter by type if provided
    role: str | None = None,       # filter by role if provided
    k: int = RAG_TOP_K,
) -> list[RetrievedChunk]:
    """
    Embed query and fetch top-k chunks from ChromaDB.
    Applies metadata filters when provided to narrow scope (saves re-ranking cost).
    """
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()

    # Build where clause — only filter when we have something specific
    where: dict = {}
    if doc_type and role and role != "all":
        where = {"$and": [{"type": {"$eq": doc_type}}, {"role": {"$in": [role, "all"]}}]}
    elif doc_type:
        where = {"type": {"$eq": doc_type}}
    elif role and role != "all":
        where = {"role": {"$in": [role, "all"]}}

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, max(1, collection.count())),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                text=doc,
                doc_type=meta.get("type", "unknown"),
                metadata=meta,
                score=dist,
            )
        )
    return chunks


def retrieve_for_question_gen(
    target_role: str,
    skills: list[str],
    experience_level: str,
    n_technical: int = 3,
    n_behavioral: int = 2,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    Fetch separate technical and behavioral chunks for question generation.
    Two focused queries are cheaper than one broad query with post-filtering.
    """
    # Map UI role name to KB role key
    role_key = _normalise_role(target_role)

    top_skills = ", ".join(skills[:5])  # cap to avoid long queries
    tech_query = f"{target_role} {top_skills} technical interview questions {experience_level}"
    behavioral_query = f"behavioral interview STAR {experience_level} {target_role}"

    tech_chunks = retrieve(tech_query, doc_type="technical", role=role_key, k=n_technical)
    beh_chunks = retrieve(behavioral_query, doc_type="behavioral", k=n_behavioral)

    return tech_chunks, beh_chunks


def retrieve_for_answer(
    question: str,
    question_type: str,
    role: str,
) -> list[RetrievedChunk]:
    """Single retrieval for model answer generation."""
    role_key = _normalise_role(role)
    doc_type = "technical" if question_type.lower() == "technical" else "behavioral"
    return retrieve(question, doc_type=doc_type, role=role_key, k=3)


def _normalise_role(role: str) -> str:
    """Map free-text role names to KB role keys."""
    role_lower = role.lower()
    if "data" in role_lower and ("sci" in role_lower or "analyst" in role_lower):
        return "data_scientist"
    if "software" in role_lower or "backend" in role_lower or "fullstack" in role_lower:
        return "software_engineer"
    if "frontend" in role_lower or "ui" in role_lower:
        return "software_engineer"
    if "ml" in role_lower or "machine learning" in role_lower:
        return "data_scientist"
    # Default — returns both by not filtering on role
    return "all"


def format_chunks_for_prompt(chunks: list[RetrievedChunk], max_chars: int = 1200) -> str:
    """
    Serialise retrieved chunks into a compact string for prompt injection.
    Hard-caps total characters to control token spend.
    """
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        entry = f"[{i}] {chunk.text.strip()}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n---\n".join(parts)
