import pandas as pd
import json
import os
import sys
from tqdm import tqdm
import bm25s
import numpy as np
from datetime import datetime, timedelta
import time
import argparse

def get_time_aware_scores(query, corpus, retriever, cutoff_time, half_life_days=30, k=3):
    query_tokens = bm25s.tokenize([query])

    retrieved = retriever.retrieve(query_tokens, k=k)
    docs = retrieved.documents[0]
    scores = retrieved.scores[0]
    scores = [x for _, x in sorted(zip(docs, scores))]

    # apply time decay
    now = cutoff_time
    decayed_scores = []

    for i, score in enumerate(scores):
        age = (now - corpus[i]["date"]).days
        decay_factor = np.exp(-np.log(2) * age / half_life_days)
        decayed_scores.append(score * decay_factor)


    return decayed_scores

def time_bm25(search_results, half_life=15):
    results_data = []
    for question_data in tqdm(search_results, desc="Processing questions"):
        question_id = question_data['id']
        question = question_data['question']
        description = question_data.get('description', '')
        cutoff_time = datetime.strptime(question_data.get('resolution_date'), "%Y-%m-%d %H:%M:%S")

        final_scores = []
        if question_data.get('web_results'):

            # get web results
            web_results = question_data['web_results']

            data = []
            web_results_out = []

            # format context
            for i, result in enumerate(web_results):
                highlights = result.get('highlights', 'No highlights available')
                if highlights:
                    highlights = highlights[0]
                else:
                    highlights = ""
                date = datetime.strptime(result.get('published_date'), "%Y-%m-%dT%H:%M:%S.%fZ")
                data.append((highlights, date))
                web_results_out

            corpus = [{'text': d[0], "date": d[1]} for d in data]

            docs = [d["text"] for d in corpus]
            tokenized_corpus = bm25s.tokenize(docs)

            # index with bm25s
            retriever = bm25s.BM25(method="bm25+")
            retriever.index(tokenized_corpus)

            final_scores = get_time_aware_scores(question, corpus, retriever, cutoff_time, k=len(corpus), half_life_days=half_life)

        if final_scores:
            for i, result in enumerate(web_results):
                web_results[i]['relevance_score'] = final_scores[i]


        web_results = sorted(web_results, key=lambda q: q['relevance_score'], reverse=True)
        question_data['web_results'] = web_results
        results_data.append(question_data)
    return results_data

def main():
    parser = argparse.ArgumentParser(description="Rerank shared evidence pools using time-aware BM25+.")
    parser.add_argument("--input", default=".\\data\\search_data.json")
    parser.add_argument("--output", default=".\\data_zhang\\results_time_aware_bm25_rerank.json")
    parser.add_argument("--halflife", default=15)
    args = parser.parse_args()
    with open(args.input, 'r', encoding="utf-8") as f:
        search_results = json.load(f)

    results_data = time_bm25(search_results, half_life=args.halflife)

    with open(args.output, "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    sys.exit(main())
