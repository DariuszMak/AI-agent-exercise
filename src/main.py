import re

import pandas as pd


def run_conversion(input_path: str) -> tuple[str, str]:
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[2:-1]
    content = "".join(lines)

    content = re.sub(r",Dariusz Makarewicz,.*?,", "", content)
    content = re.sub(r",,+", ",", content)
    content = re.sub(r"^,.*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r",(?:\d+(\.\d+)?)(?:,+)?$", "", content, flags=re.MULTILINE)
    content = re.sub(r",+$", "", content, flags=re.MULTILINE)
    content = "Project,Key,Hours\n" + content

    converted_path = f"{input_path}.converted.csv"
    with open(converted_path, "w", encoding="utf-8") as f:
        f.write(content)

    df = pd.read_csv(converted_path)

    df = df.dropna(subset=["Project", "Hours"])
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    df = df.dropna(subset=["Hours"])

    summary = df.groupby("Project", as_index=False).sum(numeric_only=True)

    total = summary["Hours"].sum()
    summary.loc[len(summary.index)] = ["Total", total]

    def format_hours(hours: float) -> str:
        h = int(hours)
        m = round((hours - h) * 60)
        return f"{h}h {m}m"

    summary["Formatted Hours"] = summary["Hours"].apply(format_hours)

    pivot_path = f"{input_path}.pivot.csv"
    summary.to_csv(pivot_path, index=False)

    return converted_path, pivot_path


if __name__ == "__main__":
    import sys

    run_conversion(sys.argv[1])
