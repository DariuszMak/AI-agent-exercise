import os
from collections.abc import Iterable

from openai import OpenAI

_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)


def generate_answer(
    question: str,
    context_chunks: Iterable[str],
    model: str = "llama3.1",
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

    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem RAG."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()
