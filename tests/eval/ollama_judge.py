from __future__ import annotations

import json
import re

from openai import OpenAI

_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
_MODEL = "gemma:2b"  # match your ollama list


def _ask_judge(prompt: str) -> dict:
    """Call Ollama and extract the first JSON object from the response."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return {"score": 0, "reason": f"no JSON found in: {raw[:200]}"}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"score": 0, "reason": f"invalid JSON: {raw[:200]}"}


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """Is the answer grounded in the retrieved context (no hallucination)?"""
    context_block = "\n---\n".join(contexts)
    prompt = f"""You are evaluating a RAG system. Decide if the answer is fully supported by the context.

CONTEXT:
{context_block}

ANSWER:
{answer}

Reply with ONLY a JSON object like this, nothing else:
{{"score": 1, "reason": "short explanation"}}

score must be 1 (faithful) or 0 (hallucination detected)."""
    return _ask_judge(prompt)


def score_answer_relevancy(question: str, answer: str) -> dict:
    """Does the answer actually address the question?"""
    prompt = f"""You are evaluating a RAG system. Decide if the answer is relevant to the question.

QUESTION:
{question}

ANSWER:
{answer}

Reply with ONLY a JSON object like this, nothing else:
{{"score": 1, "reason": "short explanation"}}

score must be 1 (relevant) or 0 (not relevant or off-topic)."""
    return _ask_judge(prompt)


def score_context_relevancy(question: str, contexts: list[str]) -> dict:
    """Did the retriever return chunks useful for answering this question?"""
    context_block = "\n---\n".join(contexts)
    prompt = f"""You are evaluating a RAG retriever.
    Decide if the retrieved context is useful for answering the question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_block}

Reply with ONLY a JSON object like this, nothing else:
{{"score": 1, "reason": "short explanation"}}

score must be 1 (context is useful) or 0 (context is irrelevant or empty)."""
    return _ask_judge(prompt)
