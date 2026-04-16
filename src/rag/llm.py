import os
from functools import lru_cache
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from collections.abc import Iterable

_CLIENT = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    timeout=30.0,
)


@lru_cache(maxsize=256)
def _cached_completion(prompt: str, model: str) -> str:
    response = _CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem RAG."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""


def generate_answer(
    question: str,
    context_chunks: Iterable[str],
    model: str = "phi3",
) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
Odpowiadaj wyłącznie na podstawie kontekstu.
Jeżeli nie ma informacji w kontekście, odpowiedz: "Nie wiem".

KONTEKST:
{context}

PYTANIE:
{question}
""".strip()

    return _cached_completion(prompt, model)
