import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


API_KEYS_PATH = Path("api_keys.json")
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen3.5-27b@q4_k_xl"
NUM_RESULTS = 20
FETCH_MULTIPLIER = 2
SEARCH_WINDOW_DAYS = 180
TOP_K = 10
TOP_LOGPROBS = 5
TEMPERATURE = 0.0

EXCLUDED_DOMAINS = [
    "polymarket.com", "kalshi.com", "manifold.markets", "metaculus.com",
    "predictit.org", "goodjudgment.com", "gjopen.com", "wikipedia.org",
    "youtube.com", "instagram.com", "electionbettingodds.com",
]

RERANK_SYSTEM = """You are evaluating evidence for a binary prediction market question.
Consider ONLY what this specific piece of evidence suggests, not your prior knowledge.
Answer with one word: YES or NO.
/no_think"""

RERANK_USER = """Question: {question}

Resolution criteria:
{description}

Evidence title: {title}
Evidence published: {published_date}
Evidence content: {text}

Based ONLY on this evidence, does it suggest the answer is YES or NO?"""

PREDICT_SYSTEM = """You are a forecaster answering a binary prediction market question from evidence only.
Answer with exactly one word: YES or NO.
/no_think"""

PREDICT_USER = """Question: {question}

Resolution criteria:
{description}

Evidence:
{evidence}

Based ONLY on the evidence above, will the answer be YES or NO?"""

EXPLAIN_SYSTEM = """You are a forecaster explaining a binary prediction market prediction from evidence only.
Give 2-3 concise sentences. Mention the main evidence themes and uncertainty if relevant.
Do not use bullet points.
/no_think"""

EXPLAIN_USER = """Question: {question}

Resolution criteria:
{description}

Top evidence:
{evidence}

Final prediction: {prediction}

Explain this prediction briefly."""

SESSION = requests.Session()


@dataclass
class ScoredDoc:
    index: int
    title: str
    url: str
    published_date: str
    text: str
    logprob_yes: float
    logprob_no: float
    z_yes: float = 0.0
    z_no: float = 0.0
    relevance_score: float = 0.0

    @property
    def direction(self) -> float:
        return self.logprob_yes - self.logprob_no

    @property
    def quadrant(self) -> str:
        if self.logprob_yes + self.logprob_no < -20:
            return "irrelevant"
        if abs(self.direction) < 1.0:
            return "contested"
        return "yes" if self.direction > 0 else "no"


def header(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def api_keys():
    keys = {}
    if API_KEYS_PATH.exists():
        with API_KEYS_PATH.open(encoding="utf-8") as f:
            keys.update(json.load(f))
    if os.getenv("EXA_API_KEY"):
        keys["EXA"] = os.getenv("EXA_API_KEY")
    return keys


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_days_ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")


def query_text(question, description):
    return f"{question.strip()}\nResolution criteria: {description.strip()[:700]}" if description.strip() else question.strip()


def normalize_exa(results):
    output = []
    for item in getattr(results, "results", results) or []:
        get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
        output.append({
            "title": get("title", "") or "",
            "url": get("url", "") or "",
            "published_date": get("published_date") or get("publishedDate") or "",
            "highlights": get("highlights", []) or [],
        })
    return output


def search(question, description, exa_key, num_results):
    # live web search
    from exa_py import Exa
    exa = Exa(exa_key)
    query = query_text(question, description)
    contents = {"highlights": {"query": query, "max_characters": 2500}}
    fetch_count = max(num_results * FETCH_MULTIPLIER, 40)
    results = exa.search(
        query,
        num_results=fetch_count,
        start_published_date=iso_days_ago(SEARCH_WINDOW_DAYS),
        end_published_date=iso_now(),
        contents=contents,
        exclude_domains=EXCLUDED_DOMAINS,
    )
    # keep dated news pages
    seen = set()
    kept = []
    for result in normalize_exa(results):
        url = result.get("url") or ""
        if not url or url in seen or not result.get("published_date"):
            continue
        if any(domain in url for domain in EXCLUDED_DOMAINS):
            continue
        seen.add(url)
        kept.append(result)
        if len(kept) >= num_results:
            break
    return kept


def extract_logprobs(response):
    yes = -100.0
    no = -100.0
    try:
        for item in response["output"]:
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    continue
                first = (content.get("logprobs") or [])[0]
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
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return yes, no


def yes_no_logprobs(system_prompt, user_prompt):
    response = SESSION.post(
        f"{LMSTUDIO_BASE_URL}/responses",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": TEMPERATURE,
            "max_output_tokens": 4,
            "include": ["message.output_text.logprobs"],
            "top_logprobs": TOP_LOGPROBS,
        },
        timeout=90,
    )
    response.raise_for_status()
    return extract_logprobs(response.json())


def text_response(system_prompt, user_prompt, max_tokens=160):
    response = SESSION.post(
        f"{LMSTUDIO_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def znorm(values):
    valid = [v for v in values if v > -90.0]
    if len(valid) < 2:
        return [0.0] * len(values)
    mean = sum(valid) / len(valid)
    std = (sum((v - mean) ** 2 for v in valid) / len(valid)) ** 0.5
    return [0.0] * len(values) if std == 0 else [(v - mean) / std for v in values]


def doc_text(result, limit=1200):
    highlights = result.get("highlights") or []
    text = " ".join(highlights) if isinstance(highlights, list) else str(highlights)
    return text.replace("\n", " ").strip()[:limit]


def score_doc(question, description, result, index):
    prompt = RERANK_USER.format(
        question=question,
        description=description or "(no additional criteria provided)",
        title=result.get("title", ""),
        published_date=result.get("published_date") or "unknown",
        text=doc_text(result, 800),
    )
    try:
        yes, no = yes_no_logprobs(RERANK_SYSTEM, prompt)
    except Exception:
        yes, no = -100.0, -100.0
    return ScoredDoc(index, result.get("title", ""), result.get("url", ""), result.get("published_date", ""), doc_text(result), yes, no)


def rerank(question, description, results):
    # score each document
    docs = [score_doc(question, description, result, i) for i, result in enumerate(results)]
    z_yes = znorm([doc.logprob_yes for doc in docs])
    z_no = znorm([doc.logprob_no for doc in docs])
    for doc, yes, no in zip(docs, z_yes, z_no):
        doc.z_yes = yes
        doc.z_no = no
        doc.relevance_score = max(yes, no)
    return sorted(docs, key=lambda doc: doc.relevance_score, reverse=True)


def evidence_text(docs, top_k, chars):
    parts = []
    for i, doc in enumerate(docs[:top_k], 1):
        parts.append(f"[{i}] {doc.title} ({doc.published_date})\nstance_z=max({doc.z_yes:+.2f}, {doc.z_no:+.2f})={doc.relevance_score:+.2f}\n{doc.text[:chars]}")
    return "\n\n".join(parts)


def predict(question, description, docs, top_k):
    # predict from top evidence
    prompt = PREDICT_USER.format(
        question=question,
        description=description or "(no additional criteria provided)",
        evidence=evidence_text(docs, min(top_k, 6), 320),
    )
    yes, no = yes_no_logprobs(PREDICT_SYSTEM, prompt)
    return ("YES" if yes >= no else "NO"), yes, no


def explain(question, description, prediction, docs, top_k):
    prompt = EXPLAIN_USER.format(
        question=question,
        description=description or "(no additional criteria provided)",
        evidence=evidence_text(docs, min(top_k, 5), 700),
        prediction=prediction,
    )
    return text_response(EXPLAIN_SYSTEM, prompt)


def first_words(text, limit=100):
    words = text.split()
    return " ".join(words[:limit]) + (" ..." if len(words) > limit else "")


def showcase(docs):
    choices = [
        ("Top YES", max([d for d in docs if d.direction > 0], key=lambda d: d.z_yes, default=None)),
        ("Top NO", max([d for d in docs if d.direction < 0], key=lambda d: d.z_no, default=None)),
        ("Most Uncertain", min([d for d in docs if d.quadrant != "irrelevant"], key=lambda d: abs(d.direction), default=None)),
    ]
    header("Showcase Evidence")
    seen = set()
    for label, doc in choices:
        if doc is None or doc.index in seen:
            continue
        seen.add(doc.index)
        print(f"[{label}] {doc.title}")
        print(f"URL: {doc.url}")
        print(f"stance_z={doc.relevance_score:+.2f} (z_yes={doc.z_yes:+.2f}, z_no={doc.z_no:+.2f}, lp_yes={doc.logprob_yes:+.2f}, lp_no={doc.logprob_no:+.2f}, quadrant={doc.quadrant})")
        print(first_words(doc.text))
        print()


def run(question, description, num_results, top_k):
    exa_key = api_keys().get("EXA")
    if not exa_key:
        print("No EXA API key found. Create api_keys.json or set EXA_API_KEY.", file=sys.stderr)
        return 1
    try:
        SESSION.get(f"{LMSTUDIO_BASE_URL}/models", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"LM Studio server not reachable: {exc}", file=sys.stderr)
        return 1

    header("Live Forecast Demo")
    print(f"Question: {question}")
    if description:
        print(f"Resolution criteria: {description[:220]}{'...' if len(description) > 220 else ''}")
    print(f"Search target: {num_results} recent web documents")
    print(f"Reranker / predictor: local stance z-score ({MODEL})")

    header("Step 1 - Search")
    results = search(question, description, exa_key, num_results)
    print(f"Collected {len(results)} filtered documents")
    if not results:
        return 1

    header("Step 2 - Rerank")
    docs = rerank(question, description, results)
    print(f"Reranked {len(docs)} documents")

    header("Step 3 - Predict")
    prediction, yes, no = predict(question, description, docs, top_k)
    print(f"Prediction: {prediction}")
    print(f"logprob_yes: {yes:+.4f}")
    print(f"logprob_no:  {no:+.4f}")
    print(f"margin:      {(yes - no):+.4f}\n")
    print(explain(question, description, prediction, docs, top_k))
    showcase(docs)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Live end-to-end forecasting demo")
    parser.add_argument("--question")
    parser.add_argument("--description", default="")
    parser.add_argument("--num-results", type=int, default=NUM_RESULTS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.question:
        return run(args.question, args.description, args.num_results, args.top_k)

    header("Live Forecast Demo")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not question or question.lower() in {"quit", "exit", "q"}:
            return 0
        description = input("Resolution criteria (optional): ").strip()
        status = run(question, description, args.num_results, args.top_k)
        if status:
            return status


if __name__ == "__main__":
    sys.exit(main())
