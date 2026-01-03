from pathlib import Path

def load_documents(path: Path):
    return [
        {"id": f.name, "text": f.read_text(encoding="utf-8")}
        for f in path.glob("*.txt")
    ]
