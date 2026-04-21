import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


INPUT_PATH = Path("data_leon/data/politics_resolved_2025_09_01_to_2026_04_05/all_sources.jsonl")
OUTPUT_PATH = Path("data/candidates_600.jsonl")
MIN_HORIZON_DAYS = 14
MIN_RESOLUTION_DATE = datetime(2026, 2, 24, tzinfo=timezone.utc)
TARGET_TOTAL = 600
SEED = 42
REQUIRED_FIELDS = ["source", "question", "resolution", "created_time", "resolution_time", "question_id"]


def parse_dt(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eligible(row, min_resolution_date, min_horizon_days):
    if any(not row.get(field) for field in REQUIRED_FIELDS):
        return False
    created = parse_dt(row.get("created_time"))
    resolved = parse_dt(row.get("resolution_time"))
    if not created or not resolved:
        return False
    if (resolved - created).total_seconds() / 86400 < min_horizon_days:
        return False
    if resolved <= min_resolution_date:
        return False
    return row.get("resolution", "").upper() in ("YES", "NO")


def stratified_sample(rows, target, seed):
    rng = random.Random(seed)
    by_month = defaultdict(list)
    for row in rows:
        by_month[parse_dt(row["resolution_time"]).strftime("%Y-%m")].append(row)

    selected = []
    per_month = max(1, target // max(1, len(by_month)))
    for month in sorted(by_month):
        pool = by_month[month]
        rng.shuffle(pool)
        selected.extend(pool[: min(per_month, len(pool))])

    selected_ids = {row["question_id"] for row in selected}
    remaining = [row for row in rows if row["question_id"] not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: target - len(selected)])
    rng.shuffle(selected)
    return selected[:target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--min-horizon", type=int, default=MIN_HORIZON_DAYS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-resolution-date", default=MIN_RESOLUTION_DATE.strftime("%Y-%m-%d"))
    args = parser.parse_args()

    min_date = datetime.strptime(args.min_resolution_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rows = load_jsonl(Path(args.input))
    rows = [row for row in rows if eligible(row, min_date, args.min_horizon)]
    selected = stratified_sample(rows, args.target, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    months = Counter(parse_dt(row["resolution_time"]).strftime("%Y-%m") for row in selected)
    sources = Counter(row["source"] for row in selected)
    print(f"Eligible: {len(rows)} | Selected: {len(selected)} -> {output}")
    print(f"By month: {dict(sorted(months.items()))}")
    print(f"By source: {dict(sources)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
