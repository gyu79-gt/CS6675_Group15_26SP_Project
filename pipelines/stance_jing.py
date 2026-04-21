import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


INPUT_PATH = Path("data/search_data.json")
OUTPUT_PATH = Path("pipeline_jing/reranked_stance_jing.jsonl")
BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen3.5-27b@q4_k_xl"
TOP_LOGPROBS = 5
TEMPERATURE = 0.0

SESSION = requests.Session()

SYSTEM_PROMPT = """You are evaluating evidence for a binary prediction market question.
Consider ONLY what this specific piece of evidence suggests, not your prior knowledge.
If the evidence is ambiguous, irrelevant, or could support either side, it is okay to be uncertain.
Answer with one word: YES or NO.
/no_think"""

USER_TEMPLATE = """Question: {question}

Evidence title: {title}
Evidence published: {published_date}
Evidence content: {text}

Based ONLY on this evidence, does it suggest the answer is YES or NO?"""


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


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def result_text(result, limit=800):
    highlights = result.get("highlights") or []
    text = " ".join(highlights) if isinstance(highlights, list) else str(highlights)
    return text.replace("\n", " ").strip()[:limit]


def extract_logprobs(response):
    # read first token
    yes = -100.0
    no = -100.0
    try:
        for item in response["output"]:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    continue
                entries = content.get("logprobs") or []
                if not entries:
                    continue
                first = entries[0]
                for entry in first.get("top_logprobs", []):
                    token = entry.get("token", "").strip().upper()
                    logprob = float(entry.get("logprob", -100.0))
                    if token in ("YES", "Y"):
                        yes = max(yes, logprob)
                    elif token in ("NO", "N"):
                        no = max(no, logprob)
                token = first.get("token", "").strip().upper()
                logprob = float(first.get("logprob", -100.0))
                if token in ("YES", "Y"):
                    yes = max(yes, logprob)
                elif token in ("NO", "N"):
                    no = max(no, logprob)
                return yes, no
    except (KeyError, TypeError, ValueError):
        pass
    return yes, no


def call_model(base_url, model, question, result):
    user_msg = USER_TEMPLATE.format(
        question=question,
        title=result.get("title") or "",
        published_date=result.get("published_date") or "unknown",
        text=result_text(result),
    )
    response = SESSION.post(
        f"{base_url}/responses",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "input": user_msg,
            "instructions": SYSTEM_PROMPT,
            "temperature": TEMPERATURE,
            "max_output_tokens": 4,
            "include": ["message.output_text.logprobs"],
            "top_logprobs": TOP_LOGPROBS,
        },
        timeout=60,
    )
    response.raise_for_status()
    return extract_logprobs(response.json())


def znorm(values):
    # compare within question
    valid = [v for v in values if v > -90.0]
    if len(valid) < 2:
        return [0.0] * len(values)
    mean = sum(valid) / len(valid)
    std = (sum((v - mean) ** 2 for v in valid) / len(valid)) ** 0.5
    if std == 0:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


def quadrant(logprob_yes, logprob_no):
    direction = logprob_yes - logprob_no
    informativeness = logprob_yes + logprob_no
    if informativeness < -20:
        return "irrelevant"
    if abs(direction) < 1.0:
        return "contested"
    return "yes" if direction > 0 else "no"


def score_documents(question, web_results, base_url, model, workers):
    # one model call per doc
    scored = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(call_model, base_url, model, question, result): (i, result)
            for i, result in enumerate(web_results)
        }
        for future in as_completed(futures):
            i, result = futures[future]
            try:
                logprob_yes, logprob_no = future.result()
            except Exception:
                logprob_yes, logprob_no = -100.0, -100.0
            scored.append(
                {
                    "index": i,
                    "logprob_yes": logprob_yes,
                    "logprob_no": logprob_no,
                    "direction": logprob_yes - logprob_no,
                    "quadrant": quadrant(logprob_yes, logprob_no),
                }
            )
    scored.sort(key=lambda x: x["index"])
    z_yes = znorm([s["logprob_yes"] for s in scored])
    z_no = znorm([s["logprob_no"] for s in scored])
    for item, yes, no in zip(scored, z_yes, z_no):
        item["z_yes"] = round(yes, 4)
        item["z_no"] = round(no, 4)
        item["relevance_score"] = round(max(yes, no), 4)
    return scored


def apply_scores(rec, scored):
    web_results = rec.get("web_results") or []
    by_index = {s["index"]: s for s in scored}
    for i, result in enumerate(web_results):
        score = by_index.get(i)
        if not score:
            result["relevance_score"] = -200.0
            continue
        result["relevance_score"] = score["relevance_score"]
        result["logprob_yes"] = round(score["logprob_yes"], 4)
        result["logprob_no"] = round(score["logprob_no"], 4)
        result["z_yes"] = score["z_yes"]
        result["z_no"] = score["z_no"]
        result["direction"] = round(score["direction"], 4)
        result["quadrant"] = score["quadrant"]
    rec["stance_summary"] = {
        "num_yes": sum(1 for s in scored if s["quadrant"] == "yes"),
        "num_no": sum(1 for s in scored if s["quadrant"] == "no"),
        "num_contested": sum(1 for s in scored if s["quadrant"] == "contested"),
        "num_irrelevant": sum(1 for s in scored if s["quadrant"] == "irrelevant"),
    }
    return rec


def rerank_records(records, base_url, model, workers):
    output = []
    for rec in tqdm(records, desc="Reranking", unit="q"):
        web_results = rec.get("web_results") or []
        if web_results:
            scored = score_documents(rec.get("question", ""), web_results, base_url, model, workers)
            rec = apply_scores(rec, scored)
        output.append(rec)
    return output


def main():
    parser = argparse.ArgumentParser(description="Z-score stance reranker using local LM Studio")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    try:
        response = requests.get(f"{args.base_url}/models", timeout=5)
        response.raise_for_status()
    except Exception as exc:
        print(f"Server not reachable: {exc}", file=sys.stderr)
        return 1

    records = load_records(Path(args.input))
    if args.max_questions:
        records = records[: args.max_questions]
    print(f"Model: {args.model}")
    print(f"Questions: {len(records)}")
    output = rerank_records(records, args.base_url, args.model, args.workers)
    write_jsonl(Path(args.output), output)
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
