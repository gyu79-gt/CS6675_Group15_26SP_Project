import json
import math
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
def tokenize(text: str):
    text = (text or "").lower()
    return TOKEN_PATTERN.findall(text)


def bm25(question: str, content: str, all_contents: list[str]):
    # lexical score
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
        deno = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf * tf * (k1 + 1) / deno

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


def min_max_normalize(values: list[float]):
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [0.0 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def cosine_similarity(vec1, vec2):
    arr1 = np.asarray(vec1, dtype=float)
    arr2 = np.asarray(vec2, dtype=float)

    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def rerank_one_question(question_text: str, web_results: list[dict], model, alpha: float = 0.5):
    # combine two signals
    if not web_results:
        return []

    all_contents = [get_highlights_text(result) for result in web_results]

    bm25_scores = [
        bm25(question_text, content, all_contents)
        for content in all_contents
    ]

    if question_text or any(all_contents):
        question_embedding = model.encode(question_text or "", convert_to_numpy=True)
        content_embeddings = model.encode(all_contents, convert_to_numpy=True)
        embedding_scores = [
            cosine_similarity(question_embedding, content_embedding)
            for content_embedding in content_embeddings
        ]
    else:
        embedding_scores = [0.0 for _ in web_results]

    normalized_bm25 = min_max_normalize(bm25_scores)
    normalized_embedding = min_max_normalize(embedding_scores)

    # weighted final score
    ranked_results = []
    for index, result in enumerate(web_results):
        final_score = ( alpha * normalized_bm25[index] + (1 - alpha) * normalized_embedding[index] )

        new_result = dict(result)
        new_result["relevance_score"] = final_score
        ranked_results.append(new_result)

    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked_results


def main(search_path: Path, output_path: Path, alpha: float = 0.5) -> None:
    search_results = load_search_results(search_path)
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    output = []

    for item in search_results:
        question_text = item.get("question", "")

        new_item = dict(item)
        new_item["web_results"] = rerank_one_question(
            question_text,
            item.get("web_results", []),
            model,
            alpha=alpha,
        )
        output.append(new_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved reranked results to {output_path}")


if __name__ == "__main__":
    main(
        Path("data/search_data.json"),
        Path("pipeline_leon/ranking_result/search_data_hybridbm25.json"),
    )
