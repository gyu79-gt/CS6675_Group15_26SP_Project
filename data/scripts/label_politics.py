import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


API_BASE = "http://127.0.0.1:1234/v1"
INPUT_PATH = Path("data/candidates_600.jsonl")
OUTPUT_PATH = Path("data/candidates_600_labeled.jsonl")
MODEL = "qwen3.5-27b@q5_k_m"
WORKERS = 8

SYSTEM_PROMPT = """You are a classifier. Given a prediction-market question and its description, decide whether the question is about POLITICS or not.
Politics includes elections, legislation, government policy, political figures, geopolitics, diplomacy, wars, sanctions, government agencies, executive orders, congressional actions, political parties, referendums, and appointments.
NOT politics includes sports, entertainment, crypto prices, tech products, science, weather, celebrity gossip, or financial markets unless directly about government economic policy.
Respond with exactly one word: politics or not_politics
/no_think"""


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_ids(path):
    return {r.get("question_id") for r in load_jsonl(path) if r.get("question_id")}


def classify(row, model):
    prompt = f"Question: {row.get('question', '')}\nDescription: {(row.get('description') or '')[:500]}"
    response = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 20,
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip().lower()
    result = dict(row)
    if "not_politics" in content or "not politics" in content:
        result["label"] = "not_politics"
    elif "politics" in content:
        result["label"] = "politics"
    else:
        result["label"] = f"unknown:{content[:50]}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.overwrite and output.exists():
        output.unlink()
    rows = load_jsonl(Path(args.input))
    done = completed_ids(output)
    pending = [r for r in rows if r.get("question_id") not in done]
    print(f"Loaded: {len(rows)} | Already labeled: {len(done)} | Pending: {len(pending)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with output.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify, row, args.model): row for row in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling", unit="q"):
            try:
                f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                f.flush()
            except Exception as exc:
                errors += 1
                print(f"Error on {futures[future].get('question_id')}: {exc}", file=sys.stderr)

    labeled = load_jsonl(output)
    politics = sum(1 for r in labeled if r.get("label") == "politics")
    not_politics = sum(1 for r in labeled if r.get("label") == "not_politics")
    print(f"Total labeled: {len(labeled)} | politics: {politics} | not_politics: {not_politics} | errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
