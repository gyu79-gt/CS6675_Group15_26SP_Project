import json
import math
import re
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
def tokenize(text: str):
    text = (text or "").lower()
    return TOKEN_PATTERN.findall(text)


def bm25(question: str, content: str, all_contents: list[str]):
    # lexical matching score
    query_tokens = tokenize(question)
    doc_tokens = tokenize(content)
    docs_tokens = [tokenize(item) for item in all_contents]

    if not query_tokens or not doc_tokens or not docs_tokens:
        return 0.0

    k1 = 1.5
    b = 0.75
    avg_doc_len = sum(len(tokens) for tokens in docs_tokens) / len(docs_tokens)
    if avg_doc_len == 0:
        return 0.0

    score = 0.0
    doc_len = len(doc_tokens)

    for term in query_tokens:
        tf = doc_tokens.count(term)
        if tf == 0:
            continue

        df = 0
        for tokens in docs_tokens:
            if term in tokens:
                df += 1

        idf = math.log(1 + (len(docs_tokens) - df + 0.5) / (df + 0.5))
        denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf * tf * (k1 + 1) / denominator

    return score

def load_search_results(search_path: Path):
    with open(search_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_highlights_text(web_result: dict):
    highlights = web_result.get("highlights", [])
    if isinstance(highlights, list):
        return " ".join(str(item) for item in highlights if item)
    if highlights:
        return str(highlights)
    return ""


def rerank_one_question(question_text: str, web_results: list[dict]):
    # rank documents
    all_contents = [get_highlights_text(result) for result in web_results]
    ranked_results = []

    for result, content in zip(web_results, all_contents):
        new_result = dict(result)
        new_result["relevance_score"] = round(bm25(question_text, content, all_contents), 6)
        ranked_results.append(new_result)

    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked_results


def main(search_path: Path, output_path: Path):
    search_results = load_search_results(search_path)
    output = []

    for item in search_results:
        question_text = item.get("question", "")

        new_item = dict(item)
        new_item["web_results"] = rerank_one_question(question_text, item.get("web_results", []))
        output.append(new_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved reranked results to {output_path}")


if __name__ == "__main__":
    main(
        Path("data/search_data.json"),
        Path("pipeline_leon/ranking_result/search_data_bm25.json"),
    )
