



def _truncate_text(text, limit=800):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_prompt(query, docs, past_answer=None, max_docs=2, max_context_chars=1400):
    selected_docs = []
    context_chars = 0

    for doc in (docs or [])[:max_docs]:
        snippet = _truncate_text(doc, limit=800)
        if not snippet:
            continue
        next_chars = context_chars + len(snippet)
        if selected_docs and next_chars > max_context_chars:
            break
        selected_docs.append(snippet)
        context_chars = next_chars

    context = "\n\n".join(selected_docs)

    if past_answer:
        prompt = f"""
    You are a helpful AI assistant. Answer using the provided context only.
    Return the shortest correct answer phrase.

Context:
{context}

Previous Answer:
{past_answer}

Question:
{query}

If the context contains the answer, use it. If not, say you don't have enough information.

Answer:
"""
    else:
        prompt = f"""
    You are a helpful AI assistant. Answer using the provided context only.
    Return the shortest correct answer phrase.

Context:
{context}

Question:
{query}

If the context contains the answer, use it. If not, say you don't have enough information.

Answer:
"""

    return prompt