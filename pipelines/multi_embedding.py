import json
import math
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class MultiFieldEmbeddingScorer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        question_weight: float = 0.35,
        description_weight: float = 0.20,
        criteria_weight: float = 0.45,
    ):
        total = question_weight + description_weight + criteria_weight

        self.model = SentenceTransformer(model_name)
        self.question_weight = question_weight
        self.description_weight = description_weight
        self.criteria_weight = criteria_weight

    @staticmethod
    def _join_text(parts):
        out = []
        for p in parts:
            if p is None:
                continue
            s = str(p).strip()
            if s:
                out.append(s)
        return " ".join(out)

    @staticmethod
    def _cosine_similarity_matrix(a, b):
        return a @ b.T

    def score_market(
        self,
        question: str,
        short_description: str,
        resolution_criteria: str,
        web_results,
        top_k=10,
        threshold=None,
    ):
        if not web_results:
            return []

        query_fields = [
            question.strip(),
            short_description.strip(),
            resolution_criteria.strip(),
        ]

        doc_texts = []
        for result in web_results:
            title = str(result.get("title", "")).strip()
            summary = result.get("summary")
            highlights = result.get("highlights", [])

            if isinstance(highlights, list):
                highlight_text = " ".join(
                    str(h).strip() for h in highlights if str(h).strip()
                )
            else:
                highlight_text = str(highlights).strip() if highlights else ""

            body_text = self._join_text([
                str(summary).strip() if summary else "",
                highlight_text,
            ])
            doc_text = self._join_text([title, body_text])
            doc_texts.append(doc_text)

        query_emb = self.model.encode(
            query_fields,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        doc_emb = self.model.encode(
            doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        sim = self._cosine_similarity_matrix(query_emb, doc_emb)
        question_scores = sim[0]
        description_scores = sim[1]
        criteria_scores = sim[2]

        final_scores = (
            self.question_weight * question_scores
            + self.description_weight * description_scores
            + self.criteria_weight * criteria_scores
        )

        ranked_indices = np.argsort(-final_scores)
        keep_indices = set(ranked_indices[:top_k].tolist())

        scored_results = []
        for i, result in enumerate(web_results):
            scored = dict(result)
            scored["question_score"] = float(question_scores[i])
            scored["description_score"] = float(description_scores[i])
            scored["criteria_score"] = float(criteria_scores[i])
            scored["relevance_score"] = float(final_scores[i])
            scored["rank"] = int(np.where(ranked_indices == i)[0][0] + 1)

            if threshold is not None:
                scored["keep"] = bool(final_scores[i] >= threshold)
            else:
                scored["keep"] = bool(i in keep_indices)

            scored_results.append(scored)

        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_results


def split_description_and_criteria(description):
    if not description or not description.strip():
        return "", ""

    text = " ".join(description.strip().split())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    criteria_keywords = [
        "resolve", "resolves", "resolved", "resolution"
    ]
    desc_sentences = []
    criteria_sentences = []
    for sent in sentences:
        s = sent.lower()
        if any(keyword in s for keyword in criteria_keywords):
            criteria_sentences.append(sent)
        else:
            desc_sentences.append(sent)

    short_description = " ".join(desc_sentences).strip()
    criteria = " ".join(criteria_sentences).strip()
    if not criteria:
        return text, text
    if not short_description:
        short_description = text

    return short_description, criteria


def process_file(
    input_path,
    output_path,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k=10,
    threshold=None,
):
    with open(input_path, "r", encoding="utf-8") as f:
        markets = json.load(f)

    scorer = MultiFieldEmbeddingScorer( #weight
        model_name=model_name,
        question_weight=0.35,
        description_weight=0.20,
        criteria_weight=0.45,
    )

    output_data = []

    for market in markets:
        question = str(market.get("question", "")).strip()
        description = str(market.get("description", "")).strip()
        short_description, criteria = split_description_and_criteria(description)

        web_results = market.get("web_results", [])
        scored_web_results = scorer.score_market(
            question=question,
            short_description=short_description,
            resolution_criteria=criteria,
            web_results=web_results,
            top_k=top_k,
            threshold=threshold,
        )

        market_out = dict(market)
        market_out["short_description"] = short_description
        market_out["resolution_criteria"] = criteria
        market_out["web_results"] = scored_web_results
        output_data.append(market_out)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_file = "data/search_data.json"
    output_file = "data/search_data_scored.json"

    process_file(
        input_path=input_file,
        output_path=output_file,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,          
        threshold=None,   
    )

    print(f"Saved scored results to: {output_file}")
