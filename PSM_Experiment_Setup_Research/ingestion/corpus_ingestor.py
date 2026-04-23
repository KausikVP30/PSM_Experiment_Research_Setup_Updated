from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


def flatten_text(value: Any) -> Iterable[str]:
    """Recursively collect text snippets from nested structures."""
    if value is None:
        return

    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return

    if isinstance(value, dict):
        for key in ("text", "sentence", "content", "description", "title", "passage"):
            if key in value:
                yield from flatten_text(value[key])
        for nested_value in value.values():
            if isinstance(nested_value, (dict, list, tuple, set)):
                yield from flatten_text(nested_value)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten_text(item)


@dataclass
class CorpusBuildResult:
    documents: List[str]
    document_count_before_chunking: int
    chunk_count: int


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 30) -> List[str]:
    """Chunk long text into overlapping word windows."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


def build_corpus_from_samples(
    samples: Sequence[Dict[str, Any]],
    min_text_len: int = 20,
    max_docs_before_chunking: int = 2500,
    max_chunks: int = 3000,
    chunk_words: int = 120,
    overlap_words: int = 30,
) -> CorpusBuildResult:
    """Build deduplicated chunked corpus from dataset records."""
    unique_docs: List[str] = []
    seen = set()

    for sample in samples:
        for field_name in ("evidence", "entity_pages", "search_results", "web_pages", "context"):
            if field_name not in sample:
                continue
            for text in flatten_text(sample[field_name]):
                if len(text) < min_text_len:
                    continue
                if text not in seen:
                    seen.add(text)
                    unique_docs.append(text)
                    if len(unique_docs) >= max_docs_before_chunking:
                        break
            if len(unique_docs) >= max_docs_before_chunking:
                break
        if len(unique_docs) >= max_docs_before_chunking:
            break

    # Fallback: include question text if evidence-like fields are sparse.
    if len(unique_docs) < 20:
        for sample in samples:
            question = str(sample.get("question", "")).strip()
            if len(question) < min_text_len:
                continue
            if question not in seen:
                seen.add(question)
                unique_docs.append(question)
                if len(unique_docs) >= max_docs_before_chunking:
                    break

    chunks: List[str] = []
    seen_chunks = set()
    for doc in unique_docs:
        for chunk in chunk_text(doc, chunk_words=chunk_words, overlap_words=overlap_words):
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break

    return CorpusBuildResult(
        documents=chunks,
        document_count_before_chunking=len(unique_docs),
        chunk_count=len(chunks),
    )
