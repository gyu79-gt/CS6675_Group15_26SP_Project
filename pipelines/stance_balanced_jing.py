import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from stance_jing import BASE_URL, INPUT_PATH, MODEL, load_records, score_documents, write_jsonl


OUTPUT_PATH = Path("pipeline_jing/reranked_stance_balanced_jing.jsonl")
BOOST = 1000.0


def side(score):
    return "yes" if score["z_yes"] >= score["z_no"] else "no"


def balanced_top_indices(scored, top_k, min_yes, min_no):
    # start from stance rank
    ranked = sorted(scored, key=lambda x: x["relevance_score"], reverse=True)
    picked = ranked[:top_k]
    pool = ranked[top_k:]

    def enforce(target: str, minimum: int) -> None:
        while sum(1 for item in picked if side(item) == target) < minimum:
            candidates = [item for item in pool if side(item) == target]
            replaceable = [item for item in picked if side(item) != target]
            if not candidates or not replaceable:
                return
            best = max(candidates, key=lambda x: x["relevance_score"])
            worst = min(replaceable, key=lambda x: x["relevance_score"])
            picked.remove(worst)
            pool.append(worst)
            pool.remove(best)
            picked.append(best)

    enforce("yes", min_yes)
    enforce("no", min_no)
    return {item["index"] for item in picked}


def apply_balanced_scores(rec, scored, top_k, min_yes, min_no):
    # boost selected docs
    floor_ids = balanced_top_indices(scored, top_k, min_yes, min_no)
    by_index = {s["index"]: s for s in scored}
    for i, result in enumerate(rec.get("web_results") or []):
        score = by_index.get(i)
        if not score:
            result["relevance_score"] = -200.0
            result["floor_selected"] = False
            continue
        stance_score = score["relevance_score"]
        result["stance_score"] = stance_score
        result["relevance_score"] = round(stance_score + BOOST, 4) if i in floor_ids else stance_score
        result["floor_selected"] = i in floor_ids
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
    rec["balance_config"] = {"top_k": top_k, "min_yes": min_yes, "min_no": min_no}
    return rec


def main():
    parser = argparse.ArgumentParser(description="Balanced z-score stance reranker using local LM Studio")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-yes", type=int, default=2)
    parser.add_argument("--min-no", type=int, default=2)
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
    for rec in tqdm(records, desc="Reranking", unit="q"):
        web_results = rec.get("web_results") or []
        if web_results:
            scored = score_documents(rec.get("question", ""), web_results, args.base_url, args.model, args.workers)
            rec = apply_balanced_scores(rec, scored, args.top_k, args.min_yes, args.min_no)
        output.append(rec)

    write_jsonl(Path(args.output), output)
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
