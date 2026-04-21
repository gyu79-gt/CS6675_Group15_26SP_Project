import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

import requests
from tqdm import tqdm


INPUT_PATH = Path("data/search_data.json")
OUTPUT_PATH = Path("pipeline_jing/reranked_judge_jing.jsonl")
BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen3.5-27b@q4_k_xl"
PASSES = 3
MAX_DOCS = 20
SNIPPET_CHARS = 260
MAX_TOKENS = 250
TEMPERATURE = 0.0

SESSION = requests.Session()

SYSTEM_PROMPT = """You are ranking evidence documents for binary forecasting.
Assign higher scores to documents that are more useful for predicting the answer to the question before the cutoff date.
Reward specificity, concrete evidence, and incremental value.
If several documents repeat the same evidence, give the strongest one the highest score and lower the redundant ones.
Return only valid JSON matching the schema.
/no_think"""


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


def normalize_title(title):
    title = title.lower().strip()
    for sep in (" | ", " - ", " -- ", " — ", " – "):
        if sep in title:
            title = title.split(sep)[0].strip()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in title).split())


def representative_score(result):
    highlights = result.get("highlights") or []
    text = " ".join(highlights) if isinstance(highlights, list) else str(highlights)
    return len(text.strip()), len(result.get("title", ""))


def dedupe_results(results):
    # group similar titles
    groups = {}
    for i, result in enumerate(results):
        groups.setdefault(normalize_title(result.get("title", "")), []).append(i)

    kept = []
    meta = {}
    for cluster_id, indices in enumerate(groups.values(), start=1):
        best = max(indices, key=lambda i: representative_score(results[i]))
        kept.append(best)
        for i in indices:
            meta[i] = {
                "dedupe_cluster_id": cluster_id,
                "dedupe_kept": i == best,
                "dedupe_representative_index": best,
                "dedupe_cluster_size": len(indices),
            }
    kept.sort()
    return kept, meta


def snippet(result, limit):
    highlights = result.get("highlights") or []
    text = " ".join(highlights) if isinstance(highlights, list) else str(highlights)
    return text.replace("\n", " ").strip()[:limit]


def build_prompt(question, results, order, snippet_chars):
    parts = [
        f"Question: {question}",
        "",
        "Score each document's usefulness for forecasting this binary question.",
        f"Return exactly {len(results)} integer scores in a JSON array called scores.",
        "Use 0=irrelevant, 1=weak, 2=somewhat useful, 3=useful, 4=highly useful.",
        "",
    ]
    for shown_idx, doc_idx in enumerate(order):
        result = results[doc_idx]
        parts.append(f"[Doc {shown_idx}] Title: {result.get('title', '')}")
        parts.append(f"Published: {result.get('published_date', 'unknown')}")
        parts.append(f"Content: {snippet(result, snippet_chars)}")
        parts.append("")
    return "\n".join(parts)


def schema(size):
    return {
        "name": "doc_scores",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "minItems": size,
                    "maxItems": size,
                    "items": {"type": "integer", "minimum": 0, "maximum": 4},
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    }


def validate_scores(scores, size):
    if not isinstance(scores, list) or len(scores) != size:
        return False
    if any(not isinstance(x, int) or x < 0 or x > 4 for x in scores):
        return False
    return len(set(scores)) > 1


def fallback_scores(size):
    return [max(0, 4 - int(i * 5 / size)) for i in range(size)]


def score_pass(base_url, model, question, results, order, snippet_chars):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, results, order, snippet_chars)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": schema(len(results))},
    }
    response = SESSION.post(f"{base_url}/chat/completions", headers={"Content-Type": "application/json"}, json=payload, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    scores = json.loads(content).get("scores")
    if not validate_scores(scores, len(results)):
        raise RuntimeError("invalid scores")
    return scores


def score_question(rec, base_url, model, passes, max_docs, snippet_chars, seed):
    # judge all docs together
    original_results = (rec.get("web_results") or [])[:max_docs]
    if len(original_results) < max_docs:
        return [], {"validation_status": "skipped_too_few_docs"}

    kept_indices, dedupe_meta = dedupe_results(original_results)
    kept_results = [original_results[i] for i in kept_indices]
    stable_seed = int(hashlib.sha256(f"{seed}:{rec.get('id', '')}".encode()).hexdigest()[:8], 16)
    rng = random.Random(stable_seed)
    scores_by_doc = [[] for _ in kept_results]
    errors = []

    for pass_id in range(passes):
        order = list(range(len(kept_results)))
        if pass_id:
            rng.shuffle(order)
        try:
            scores = score_pass(base_url, model, rec.get("question", ""), kept_results, order, snippet_chars)
        except Exception as exc:
            errors.append(str(exc))
            scores = fallback_scores(len(kept_results))
        for shown_idx, score in enumerate(scores):
            original_doc_idx = order[shown_idx]
            scores_by_doc[original_doc_idx].append(score)

    scored_by_original = {}
    for local_idx, result in enumerate(kept_results):
        original_idx = kept_indices[local_idx]
        pass_scores = scores_by_doc[local_idx]
        scored_by_original[original_idx] = {
            "relevance_score": round(sum(pass_scores) / len(pass_scores), 4),
            "judge_scores_per_pass": pass_scores,
            "judge_score_std": round(statistics.pstdev(pass_scores), 4) if len(pass_scores) > 1 else 0.0,
        }

    scored = []
    for i, result in enumerate(original_results):
        item = scored_by_original.get(i, {"relevance_score": -1.0, "judge_scores_per_pass": [], "judge_score_std": 0.0})
        item["index"] = i
        item["result"] = result
        item.update(dedupe_meta.get(i, {}))
        scored.append(item)

    return scored, {
        "validation_status": "fallback" if errors else "ok",
        "errors": errors,
        "passes": passes,
        "original_docs": len(original_results),
        "deduped_docs": len(kept_results),
        "dropped_duplicates": len(original_results) - len(kept_results),
    }


def apply_scores(rec, scored, meta):
    # write scores back
    by_index = {item["index"]: item for item in scored}
    for i, result in enumerate(rec.get("web_results") or []):
        item = by_index.get(i)
        if not item:
            result["relevance_score"] = -1.0
            continue
        result["relevance_score"] = item["relevance_score"]
        result["judge_scores_per_pass"] = item["judge_scores_per_pass"]
        result["judge_score_std"] = item["judge_score_std"]
        result["dedupe_cluster_id"] = item.get("dedupe_cluster_id")
        result["dedupe_kept"] = item.get("dedupe_kept", True)
        result["dedupe_representative_index"] = item.get("dedupe_representative_index")
        result["dedupe_cluster_size"] = item.get("dedupe_cluster_size", 1)
    rec["judge_summary"] = meta
    return rec


def main():
    parser = argparse.ArgumentParser(description="Listwise judge reranker using local LM Studio")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--max-docs", type=int, default=MAX_DOCS)
    parser.add_argument("--snippet-chars", type=int, default=SNIPPET_CHARS)
    parser.add_argument("--seed", type=int, default=0)
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

    output = []
    fallback_count = 0
    for rec in tqdm(records, desc="Judging", unit="q"):
        scored, meta = score_question(rec, args.base_url, args.model, args.passes, args.max_docs, args.snippet_chars, args.seed)
        if meta.get("validation_status") != "ok":
            fallback_count += 1
        output.append(apply_scores(rec, scored, meta))

    write_jsonl(Path(args.output), output)
    print(f"Fallback/skipped questions: {fallback_count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
