import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


API_KEYS_PATH = Path("api_keys.json")
TOP_K = 10
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "gpt-5.4-mini", 4),
    "lmstudio": ("http://127.0.0.1:1234/v1", "qwen3.5-27b@q4_k_xl", 1),
}
THREAD_LOCAL = threading.local()

SYSTEM_PROMPT = """You are a forecaster answering binary prediction market questions.
You will be given a question and evidence documents.
Based on the evidence provided, predict whether the answer is YES or NO.
You must answer with exactly one word: YES or NO.
/no_think"""

USER_TEMPLATE = """Question: {question}

{description}

Evidence:
{evidence}

Based on the evidence above, will the answer to this question be YES or NO?"""


def session():
    s = getattr(THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        THREAD_LOCAL.session = s
    return s


def openai_key():
    for key in ("OPENAI", "OPENAI_API_KEY"):
        if os.getenv(key):
            return os.getenv(key)
    if API_KEYS_PATH.exists():
        with API_KEYS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("OPENAI") or data.get("OPENAI_API_KEY")
    return None


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_records(path):
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return load_jsonl(path)


def top_evidence(rec, top_k):
    # use reranker scores
    results = rec.get("web_results") or []
    return sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)[:top_k]


def format_evidence(results):
    parts = []
    for i, result in enumerate(results, 1):
        highlights = result.get("highlights") or []
        text = " ".join(highlights) if isinstance(highlights, list) else str(highlights)
        parts.append(f"[{i}] {result.get('title', 'Untitled')} ({result.get('published_date', 'unknown')})\n{text[:1500]}")
    return "\n\n".join(parts)


def predict(base_url, model, api_key, rec, top_k):
    # ask for one word
    evidence = format_evidence(top_evidence(rec, top_k))
    user_msg = USER_TEMPLATE.format(
        question=rec.get("question", ""),
        description=rec.get("description") or "(no description)",
        evidence=evidence or "(no evidence)",
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_completion_tokens" if api_key else "max_tokens": 50,
    }
    response = session().post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip().upper()
    if "YES" in content:
        return "YES"
    if "NO" in content:
        return "NO"
    return f"UNKNOWN:{content[:20]}"


def predict_record(base_url, model, api_key, rec, top_k):
    try:
        prediction = predict(base_url, model, api_key, rec, top_k)
    except Exception as exc:
        prediction = f"ERROR:{exc}"
    resolution = (rec.get("resolution") or "").upper()
    return {
        "id": rec.get("id", ""),
        "question": (rec.get("question") or "")[:120],
        "resolution": resolution,
        "prediction": prediction,
        "correct": prediction == resolution,
        "num_evidence": len(top_evidence(rec, top_k)),
    }


def write_results(output_path, rows, summary):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def summarize(args, model, rows):
    # count valid predictions
    valid = [r for r in rows if not r["prediction"].startswith("ERROR")]
    correct = sum(1 for r in valid if r["correct"])
    yes_res = sum(1 for r in rows if r["resolution"] == "YES")
    no_res = sum(1 for r in rows if r["resolution"] == "NO")
    yes_correct = sum(1 for r in rows if r["resolution"] == "YES" and r["correct"])
    no_correct = sum(1 for r in rows if r["resolution"] == "NO" and r["correct"])
    total = len(valid)
    return {
        "input": args.input,
        "provider": args.provider,
        "model": model,
        "top_k": args.top_k,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / max(total, 1) * 100, 2),
        "errors": len(rows) - total,
        "yes_predictions": sum(1 for r in rows if r["prediction"] == "YES"),
        "no_predictions": sum(1 for r in rows if r["prediction"] == "NO"),
        "yes_resolutions": yes_res,
        "no_resolutions": no_res,
        "yes_accuracy": round(yes_correct / max(yes_res, 1) * 100, 2),
        "no_accuracy": round(no_correct / max(no_res, 1) * 100, 2),
    }


def print_summary(summary, output_path):
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Input:      {summary['input']}")
    print(f"  Model:      {summary['model']}")
    print(f"  Top-k:      {summary['top_k']}")
    print(f"  Questions:  {summary['total']}")
    print(f"  Correct:    {summary['correct']}")
    print(f"  Accuracy:   {summary['accuracy']:.1f}%")
    if summary["errors"]:
        print(f"  Errors:     {summary['errors']}")
    print(f"\n  Predictions: {summary['yes_predictions']} YES / {summary['no_predictions']} NO")
    print(f"  Resolutions: {summary['yes_resolutions']} YES / {summary['no_resolutions']} NO")
    print("=" * 50)
    print(f"  YES questions: {summary['yes_accuracy']:.1f}%")
    print(f"  NO questions:  {summary['no_accuracy']:.1f}%")
    print(f"\n  Per-question results: {output_path}")
    print(f"  Summary: {output_path.with_suffix('.summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reranked evidence with a downstream LLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--provider", choices=PROVIDERS.keys(), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    base_url, default_model, default_workers = PROVIDERS[args.provider]
    model = args.model or default_model
    api_key = openai_key() if args.provider == "openai" else None
    workers = args.workers or default_workers
    if args.provider == "openai" and not api_key:
        print("No OpenAI API key found.", file=sys.stderr)
        return 1

    records = load_records(Path(args.input))
    records = [r for r in records if not r.get("short") and (r.get("resolution") or "").upper() in ("YES", "NO")]
    if args.max_questions:
        records = records[: args.max_questions]
    print(f"Provider: {args.provider} | Model: {model}")
    print(f"Top-k evidence: {args.top_k}")
    print(f"Workers: {workers}")
    print(f"Questions to evaluate: {len(records)}")

    rows = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(predict_record, base_url, model, api_key, rec, args.top_k) for rec in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="q"):
                rows.append(future.result())
    else:
        for rec in tqdm(records, desc="Evaluating", unit="q"):
            rows.append(predict_record(base_url, model, api_key, rec, args.top_k))

    output_path = Path(args.output or f"eval/results_{Path(args.input).stem}_top{args.top_k}.jsonl")
    summary = summarize(args, model, rows)
    write_results(output_path, rows, summary)
    print_summary(summary, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
