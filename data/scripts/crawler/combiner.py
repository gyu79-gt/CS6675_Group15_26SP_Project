import json
from datetime import datetime
from pathlib import Path


START = datetime.fromisoformat("2025-09-01")
END = datetime.fromisoformat("2026-04-05")
INPUTS = ["kalshi.jsonl", "polymarket.jsonl", "manifold.jsonl"]
OUTPUT = Path("all_sources.jsonl")


def parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000)
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T").replace("+00:00", "").replace("Z", ""))
    except ValueError:
        return None


def valid(row):
    resolved = parse_time(row.get("resolution_time"))
    return bool(resolved and row.get("close_time") and START <= resolved <= END)


def main():
    rows = []
    for name in INPUTS:
        path = Path(name)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows = [row for row in rows if valid(row)]
    rows.sort(key=lambda row: parse_time(row.get("resolution_time")) or datetime.min)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Written {len(rows)} records")


if __name__ == "__main__":
    main()
