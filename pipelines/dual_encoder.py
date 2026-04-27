import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from urllib.parse import urlparse


def rerank_p_embed(question, docs, config=None):
    # chunk then embed
    config = config or {}
    model_name = config.get("model_name", "all-MiniLM-L6-v2")
    chunk_size = int(config.get("chunk_size", 350))
    overlap = int(config.get("overlap", 50))
    max_chunks = int(config.get("max_chunks", 20))
    batch_size = int(config.get("batch_size", 64))
    top_k = config.get("top_k")
    cutoff = config.get("forecast_cutoff_time")

    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))
    model = SentenceTransformer(model_name)

    question = " ".join(str(question).split())
    q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]

    doc_scores = []
    for doc in docs:
        full_text = ""
        for key in ["full_text", "text", "content", "page_text", "article_text", "body"]:
            val = doc.get(key, "")
            if isinstance(val, list):
                val = "\n".join(str(v) for v in val if str(v).strip())
            val = str(val).strip()
            if val:
                full_text = val
                break

        if not full_text:
            parts = []
            for key in ["title"]:
                val = str(doc.get(key, "")).strip()
                if val:
                    parts.append(val)
            for key in ["highlights", "highlight"]:
                val = doc.get(key, [])
                if isinstance(val, list):
                    val = " ".join(str(v).strip() for v in val if str(v).strip())
                else:
                    val = str(val).strip()
                if val:
                    parts.append(val)
            for key in ["summary", "snippet", "description"]:
                val = str(doc.get(key, "")).strip()
                if val:
                    parts.append(val)
            full_text = " ".join(parts)

        if not full_text:
            doc_scores.append(-1.0)
            continue

        words = full_text.split()
        if len(words) <= chunk_size:
            texts = [full_text]
        else:
            step = max(1, chunk_size - max(0, min(overlap, chunk_size - 1)))
            texts = []
            for start in range(0, len(words), step):
                texts.append(" ".join(words[start:start + chunk_size]))
                if len(texts) >= max_chunks or start + chunk_size >= len(words):
                    break

        chunk_embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        sims = chunk_embs @ q_emb
        # best chunk wins
        doc_scores.append(float(np.max(sims)))

    reranked = []
    for i, doc in enumerate(docs):
        d = dict(doc)
        d["retrieval_rank"] = d.get("retrieval_rank", i + 1)
        if not str(d.get("doc_id", "")).strip():
            d["doc_id"] = str(d.get("url", "")).strip() or f"retrieval:{d['retrieval_rank']}"
        d["rerank_score"] = round(doc_scores[i], 6)
        d["relevance_score"] = round(doc_scores[i], 6)
        d["used_full_text"] = any(str(doc.get(k, "")).strip() for k in ["full_text", "text", "content"])

        if cutoff:
            try:
                cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
                pub_val = str(doc.get("published_at", doc.get("published_date", ""))).strip()
                if pub_val:
                    pub_dt = datetime.fromisoformat(pub_val.replace("Z", "+00:00"))
                    d["is_pre_cutoff"] = pub_dt <= cutoff_dt
            except ValueError:
                pass

        url = str(doc.get("url", "")).strip()
        if url:
            d["domain"] = urlparse(url).netloc
        reranked.append(d)

    reranked.sort(key=lambda d: d["rerank_score"], reverse=True)
    for rank, doc in enumerate(reranked, start=1):
        doc["rerank_rank"] = rank
        if top_k is not None:
            doc["selected_for_top_k"] = rank <= int(top_k)

    return reranked


rerank_embed = rerank_p_embed
