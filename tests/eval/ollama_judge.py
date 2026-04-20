"""
LLM-as-judge evaluation using your local Ollama instance.

Improvements over the original
-------------------------------
- Scores are now 0.0-1.0 floats instead of binary 0/1, giving more signal.
- Prompts explicitly ask for a score between 0 and 1 with one decimal place.
- A minimum model size guard is printed when gemma:2b is detected, since
  small models are unreliable judges.
- Each scorer returns a JudgeResult with score, reason, and the raw response
  for debugging.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypedDict

from openai import OpenAI

logger = logging.getLogger(__name__)

_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma:2b")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

if "2b" in _JUDGE_MODEL or "mini" in _JUDGE_MODEL.lower():
    logger.warning(
        "Judge model '%s' is small and may produce unreliable scores. "
        "Consider setting JUDGE_MODEL to a larger model such as llama3, "
        "mistral, or an API model for more consistent evaluation.",
        _JUDGE_MODEL,
    )

_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


class JudgeResult(TypedDict):
    score: float  # 0.0 – 1.0
    reason: str
    raw: str  # full model output, useful for debugging


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ask_judge(prompt: str) -> JudgeResult:
    response = _client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = (response.choices[0].message.content or "").strip()

    # Try to parse a JSON object anywhere in the response
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        logger.debug("Judge returned no JSON. Raw: %s", raw[:300])
        return {"score": 0.0, "reason": f"no JSON found in: {raw[:200]}", "raw": raw}

    try:
        parsed = json.loads(match.group())
        score = float(parsed.get("score", 0.0))
        # Normalise: if the model returned 0 or 1 integer, keep as float
        if score > 1.0:
            score = score / 10.0  # handle accidental 0-10 scale
        return {
            "score": max(0.0, min(1.0, score)),
            "reason": str(parsed.get("reason", "")),
            "raw": raw,
        }
    except json.JSONDecodeError, ValueError:
        return {"score": 0.0, "reason": f"invalid JSON: {raw[:200]}", "raw": raw}


# ---------------------------------------------------------------------------
# Public scorers
# ---------------------------------------------------------------------------


def score_faithfulness(answer: str, contexts: list[str]) -> JudgeResult:
    """
    Is every factual claim in *answer* supported by *contexts*?

    Score 1.0 = fully grounded, 0.0 = clear hallucination.
    """
    context_block = "\n---\n".join(contexts)

    prompt = f"""You are a strict RAG evaluation judge.

Task: Decide whether EVERY claim in the ANSWER is directly supported by
the CONTEXT below. Do not use outside knowledge.

CONTEXT:
{context_block}

ANSWER:
{answer}

Scoring rules:
- 1.0  : every claim is explicitly supported by the context
- 0.5  : most claims are supported but one minor detail is not
- 0.0  : the answer contains information not present in the context (hallucination)

Reply with ONLY a JSON object, nothing else:
{{"score": 0.0, "reason": "one sentence explanation"}}"""

    return _ask_judge(prompt)


def score_answer_relevancy(question: str, answer: str) -> JudgeResult:
    """
    Does *answer* directly address *question*?

    Score 1.0 = fully relevant, 0.0 = off-topic or no answer.
    """
    prompt = f"""You are a RAG evaluation judge.

Task: Decide how well the ANSWER addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Scoring rules:
- 1.0  : the answer directly and completely addresses the question
- 0.5  : the answer is partially relevant but misses key aspects
- 0.0  : the answer is off-topic, refuses to answer, or says only "I don't know"

Reply with ONLY a JSON object, nothing else:
{{"score": 0.0, "reason": "one sentence explanation"}}"""

    return _ask_judge(prompt)


def score_context_relevancy(question: str, contexts: list[str]) -> JudgeResult:
    """
    Are the retrieved *contexts* useful for answering *question*?

    Score 1.0 = highly relevant, 0.0 = irrelevant.
    """
    context_block = "\n---\n".join(contexts)

    prompt = f"""You are a RAG evaluation judge evaluating a retriever.

Task: Decide whether the RETRIEVED CONTEXT contains information that would
help answer the QUESTION.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_block}

Scoring rules:
- 1.0  : the context contains clear, direct information relevant to the question
- 0.5  : the context is tangentially related but incomplete
- 0.0  : the context is irrelevant or empty

Reply with ONLY a JSON object, nothing else:
{{"score": 0.0, "reason": "one sentence explanation"}}"""

    return _ask_judge(prompt)


def score_completeness(question: str, answer: str, ground_truth: str) -> JudgeResult:
    """
    How complete is *answer* compared to the *ground_truth* reference?

    This requires a known reference answer.  Score 1.0 = all key facts
    present, 0.0 = important facts missing.
    """
    prompt = f"""You are a RAG evaluation judge.

Task: Compare the ANSWER to the REFERENCE ANSWER and decide how many
key facts from the reference are present in the answer.

QUESTION:
{question}

REFERENCE ANSWER:
{ground_truth}

ANSWER:
{answer}

Scoring rules:
- 1.0  : all key facts from the reference are present in the answer
- 0.5  : roughly half the key facts are present
- 0.0  : most key facts are missing

Reply with ONLY a JSON object, nothing else:
{{"score": 0.0, "reason": "one sentence explanation"}}"""

    return _ask_judge(prompt)
