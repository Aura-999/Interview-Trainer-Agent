"""
kb_builder.py — Offline knowledge-base indexer.
Run once (or when KB data changes) to embed and persist all documents.

Usage:
    python backend/kb_builder.py
"""

import json
import os
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
KB_ROOT = Path("./knowledge_base")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, ~80 MB, runs locally — no API cost

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def stable_id(*parts: str) -> str:
    """Short stable ID derived from content hash — avoids duplicate inserts."""
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def ingest_technical_jsonl(
    collection: chromadb.Collection,
    filepath: Path,
    role: str,
) -> int:
    embedder = get_embedder()
    count = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Compact text — fewer tokens, still semantically complete
            text = f"Q: {item['question']}\nA: {item['answer']}"
            doc_id = stable_id(role, item["question"])

            # Skip if already indexed
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                continue

            embedding = embedder.encode(text, normalize_embeddings=True).tolist()
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[
                    {
                        "role": role,
                        "type": "technical",
                        "difficulty": item.get("difficulty", "medium"),
                        "skill_tag": item.get("skill_tag", "general"),
                    }
                ],
                ids=[doc_id],
            )
            count += 1
    return count


def ingest_behavioral_json(
    collection: chromadb.Collection,
    filepath: Path,
) -> int:
    embedder = get_embedder()
    count = 0
    with open(filepath, encoding="utf-8") as f:
        scenarios = json.load(f)

    for s in scenarios:
        # Single compact block per scenario — one retrieval unit = one STAR answer
        text = (
            f"Competency: {s['competency']}\n"
            f"Q: {s['question']}\n"
            f"S: {s['situation']}\n"
            f"T: {s['task']}\n"
            f"A: {s['action']}\n"
            f"R: {s['result']}"
        )
        doc_id = stable_id("behavioral", s["competency"], s["question"])

        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        embedding = embedder.encode(text, normalize_embeddings=True).tolist()
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[
                {
                    "type": "behavioral",
                    "competency": s["competency"],
                    "level": s.get("level", "mid"),
                    "role": "all",
                }
            ],
            ids=[doc_id],
        )
        count += 1
    return count


def ingest_text_file(
    collection: chromadb.Collection,
    filepath: Path,
    category: str,
    chunk_size: int = 300,
) -> int:
    """Paragraph-level chunking for plain-text HR guidelines."""
    embedder = get_embedder()
    text = filepath.read_text(encoding="utf-8")
    # Split on blank lines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

    # Merge short paragraphs to avoid tiny chunks
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) < chunk_size * 4:  # ~chars, not tokens
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)

    count = 0
    for i, chunk in enumerate(chunks):
        doc_id = stable_id("hr", category, str(i))
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue
        embedding = embedder.encode(chunk, normalize_embeddings=True).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"type": "hr_guideline", "category": category, "role": "all"}],
            ids=[doc_id],
        )
        count += 1
    return count


def build_index() -> None:
    print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="interview_kb",
        metadata={"hnsw:space": "cosine"},  # cosine similarity for embeddings
    )

    total = 0

    # ── Technical JSONL files ──────────────────────────────────
    tech_dir = KB_ROOT / "technical"
    role_map = {
        "software_engineer.jsonl": "software_engineer",
        "data_scientist.jsonl": "data_scientist",
    }
    for filename, role in role_map.items():
        path = tech_dir / filename
        if path.exists():
            n = ingest_technical_jsonl(collection, path, role)
            print(f"  [{role}] indexed {n} new items")
            total += n

    # ── Behavioral STAR scenarios ──────────────────────────────
    behavioral_path = KB_ROOT / "behavioral" / "star_scenarios.json"
    if behavioral_path.exists():
        n = ingest_behavioral_json(collection, behavioral_path)
        print(f"  [behavioral] indexed {n} new items")
        total += n

    # ── HR guidelines ──────────────────────────────────────────
    hr_dir = KB_ROOT / "hr_guidelines"
    for txt_file in hr_dir.glob("*.txt"):
        n = ingest_text_file(collection, txt_file, category=txt_file.stem)
        print(f"  [hr:{txt_file.stem}] indexed {n} new items")
        total += n

    print(f"\nDone. Total new items indexed: {total}")
    print(f"Collection size: {collection.count()} documents")


if __name__ == "__main__":
    build_index()
