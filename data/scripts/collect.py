import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tqdm import tqdm


CANDIDATES_PATH = Path("data/candidates_politics_586.jsonl")
EXISTING_DATA_PATH = Path("data/search_data_586.jsonl")
OUTPUT_PATH = Path("data/search_data_final.jsonl")
API_KEYS_PATH = Path("api_keys.json")
TARGET_RESULTS = 20
FETCH_MULTIPLIER = 2
DATE_WINDOW_DAYS = 180
MAX_LLM_CANDIDATES = 40
BUFFER_HOURS = 168
REQUEST_SLEEP = 0.25
LLM_API_BASE = "http://127.0.0.1:1234/v1"
LLM_MODEL = "qwen3.5-27b@q5_k_m"
LLM_WORKERS = 4

EXCLUDED_DOMAINS = [
    "polymarket.com", "kalshi.com", "manifold.markets", "metaculus.com",
    "predictit.org", "goodjudgment.com", "gjopen.com", "wikipedia.org",
    "youtube.com", "instagram.com", "ballotpedia.org", "electionbettingodds.com",
    "aol.com", "inkl.com", "polymarketintel.com", "almanac.ai", "frenflow.com",
]

LEAK_SYSTEM = """You are an evidence auditor for a forecasting study.
Given a prediction market question, correct answer, evidence cutoff date, and web result, decide whether the result reveals or strongly implies the correct answer to this specific question.
"leak" means the article directly reports the specific outcome. "safe" means background, polls, analysis, or a similar but different event.
Pay close attention to dates.
Respond in exactly two lines:
reason: <one sentence>
verdict: <leak or safe>
/no_think"""

LEAK_USER = """Question: {question}
Correct answer: {resolution}
Evidence cutoff: {cutoff}
Article published date: {published_date}

Article title: {title}
Article highlights: {highlights}"""


def parse_iso(value):
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def exa_key():
    if os.getenv("EXA_API_KEY"):
        return os.getenv("EXA_API_KEY")
    if API_KEYS_PATH.exists():
        with API_KEYS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("EXA") or data.get("EXA_API_KEY")
    return None


def effective_cutoff(row, buffer_hours):
    # leave time buffer
    dates = [dt for dt in [parse_iso(row.get("close_time")), parse_iso(row.get("resolution_time"))] if dt]
    if not dates:
        return row.get("forecast_cutoff_time") or row.get("resolution_time")
    cutoff = min(dates) - timedelta(hours=buffer_hours)
    created = parse_iso(row.get("created_time"))
    if created and cutoff <= created:
        cutoff = created
    return iso(cutoff)


def make_records(candidates, existing, buffer_hours):
    by_id = {row.get("id"): row for row in existing if row.get("id")}
    records = []
    for cand in candidates:
        qid = cand.get("question_id")
        rec = by_id.get(qid)
        cutoff = effective_cutoff(cand, buffer_hours)
        if rec:
            rec.setdefault("effective_cutoff_time", cutoff)
        else:
            resolved = parse_iso(cand.get("resolution_time"))
            rec = {
                "id": qid,
                "question": cand.get("question"),
                "description": cand.get("description"),
                "resolution": cand.get("resolution"),
                "resolution_timestamp": int(resolved.timestamp() * 1000) if resolved else None,
                "resolution_date": resolved.strftime("%Y-%m-%d %H:%M:%S") if resolved else None,
                "close_time": cand.get("close_time"),
                "forecast_cutoff_time": cand.get("forecast_cutoff_time"),
                "effective_cutoff_time": cutoff,
                "web_results": [],
            }
        records.append(rec)
    return records


def valid_result(result, cutoff):
    # enforce cutoff date
    url = result.get("url") or ""
    published = result.get("published_date")
    if not url or not published or published == "None":
        return False
    if cutoff and published > cutoff:
        return False
    return not any(domain in url for domain in EXCLUDED_DOMAINS)


def dedupe(results, cutoff, limit):
    seen = set()
    kept = []
    for result in results:
        url = result.get("url")
        if url and url not in seen and valid_result(result, cutoff):
            seen.add(url)
            kept.append(result)
        if len(kept) >= limit:
            break
    return kept


def normalize_exa(results):
    rows = []
    for item in getattr(results, "results", results) or []:
        get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
        rows.append({
            "title": get("title", ""),
            "url": get("url", ""),
            "published_date": get("published_date") or get("publishedDate"),
            "highlights": get("highlights", []) or [],
            "summary": get("summary", "") or "",
        })
    return rows


def highlight_query(rec):
    question = (rec.get("question") or "").strip()
    description = (rec.get("description") or "").strip()
    return f"{question}\nResolution criteria: {description[:700]}" if description else question


def fetch_exa(exa, rec, needed):
    cutoff = rec.get("effective_cutoff_time")
    cutoff_dt = parse_iso(cutoff)
    start = iso(cutoff_dt - timedelta(days=DATE_WINDOW_DAYS)) if cutoff_dt else None
    query = highlight_query(rec)
    results = exa.search(
        rec.get("question", ""),
        num_results=max(needed * FETCH_MULTIPLIER, 20),
        start_published_date=start,
        end_published_date=cutoff,
        contents={"highlights": {"query": query, "max_characters": 2500}},
        exclude_domains=EXCLUDED_DOMAINS,
    )
    return normalize_exa(results)


def highlights(result, limit=1000):
    text = result.get("highlights") or []
    text = " ".join(text) if isinstance(text, list) else str(text)
    return text.replace("\n", " ").strip()[:limit]


def check_leak(rec, result, model):
    prompt = LEAK_USER.format(
        question=rec.get("question", ""),
        resolution=rec.get("resolution", ""),
        cutoff=rec.get("effective_cutoff_time", ""),
        published_date=result.get("published_date", "unknown"),
        title=result.get("title", ""),
        highlights=highlights(result),
    )
    response = requests.post(
        f"{LLM_API_BASE}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "system", "content": LEAK_SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 80,
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip().lower()
    for line in reversed(content.splitlines()):
        if line.startswith("verdict:"):
            verdict = line.split(":", 1)[1].strip()
            if "leak" in verdict:
                return "leak"
            if "safe" in verdict:
                return "safe"
    return "leak" if "leak" in content else "safe"


def leak_filter(rec, candidates, model, workers):
    # remove answer leaks
    safe = []
    leaks = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_leak, rec, result, model): result for result in candidates}
        for future in as_completed(futures):
            result = futures[future]
            try:
                verdict = future.result()
            except Exception:
                verdict = "safe"
            if verdict == "leak":
                leaks += 1
            else:
                safe.append(result)
    return safe, leaks


def collect_for_record(exa, rec, args):
    rec["web_results"] = dedupe(rec.get("web_results", []), rec.get("effective_cutoff_time"), args.target)
    needed = args.target - len(rec["web_results"])
    if needed <= 0:
        return 0, 0
    existing = {r.get("url") for r in rec["web_results"] if r.get("url")}
    candidates = []
    for result in fetch_exa(exa, rec, needed):
        url = result.get("url")
        if url and url not in existing and valid_result(result, rec.get("effective_cutoff_time")):
            candidates.append(result)
            existing.add(url)
        if len(candidates) >= args.max_llm_candidates:
            break
    if args.skip_llm:
        safe, leaks = candidates, 0
    else:
        safe, leaks = leak_filter(rec, candidates, args.model, args.llm_workers)
    rec["web_results"].extend(safe[:needed])
    if len(rec["web_results"]) < args.target:
        rec["short"] = True
    return len(safe[:needed]), leaks


def main():
    parser = argparse.ArgumentParser(description="Collect Exa evidence and remove answer leaks")
    parser.add_argument("--candidates", default=str(CANDIDATES_PATH))
    parser.add_argument("--existing", default=str(EXISTING_DATA_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--target", type=int, default=TARGET_RESULTS)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--llm-workers", type=int, default=LLM_WORKERS)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--max-llm-candidates", type=int, default=MAX_LLM_CANDIDATES)
    parser.add_argument("--buffer-hours", type=int, default=BUFFER_HOURS)
    args = parser.parse_args()

    records = make_records(load_jsonl(Path(args.candidates)), load_jsonl(Path(args.existing)), args.buffer_hours)
    output = Path(args.output)
    done = {row.get("id") for row in load_jsonl(output) if row.get("id")}
    pending = [row for row in records if row.get("id") not in done]
    if args.max_questions:
        pending = pending[: args.max_questions]
    print(f"Records: {len(records)} | Already done: {len(done)} | Pending: {len(pending)}")

    key = exa_key()
    if not key:
        print("No Exa API key found", file=sys.stderr)
        return 1
    from exa_py import Exa
    exa = Exa(key)

    if not args.skip_llm:
        try:
            requests.get(f"{LLM_API_BASE}/models", timeout=5).raise_for_status()
        except Exception as exc:
            print(f"LLM server not reachable: {exc}", file=sys.stderr)
            return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"added": 0, "leaks": 0, "short": 0}
    with output.open("a", encoding="utf-8") as f:
        for rec in tqdm(pending, desc="Collecting", unit="q"):
            if args.check_existing and not args.skip_llm:
                rec["web_results"], leaks = leak_filter(rec, rec.get("web_results", []), args.model, args.llm_workers)
                stats["leaks"] += leaks
            added, leaks = collect_for_record(exa, rec, args)
            stats["added"] += added
            stats["leaks"] += leaks
            stats["short"] += 1 if rec.get("short") else 0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(REQUEST_SLEEP)

    print(f"Added: {stats['added']} | Leaks removed: {stats['leaks']} | Short: {stats['short']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
