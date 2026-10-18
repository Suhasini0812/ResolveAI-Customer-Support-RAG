"""Hybrid retriever: BM25 (keyword) + FAISS (semantic), fused with EnsembleRetriever.

Why hybrid: BM25 nails exact-token queries ("2FA", "GDPR export", error strings).
Embeddings handle paraphrase ("my login codes don't work" -> 2FA reset ticket).
EnsembleRetriever uses Reciprocal Rank Fusion to combine the two ranked lists.

We also expose a separate `score_query(query)` helper that returns the FAISS
cosine similarity between the query and each candidate doc. RRF scores are
rank-based and aren't directly interpretable as confidence, so we use FAISS
similarity as the numeric signal that feeds the confidence gate in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from knowledge_base import load_documents

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6


@dataclass
class HybridRetriever:
    ensemble: EnsembleRetriever
    vectorstore: FAISS
    embeddings: HuggingFaceEmbeddings
    documents: List[Document]

    def retrieve(self, query: str) -> List[Document]:
        return self.ensemble.invoke(query)

    def score_query(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        """Return docs paired with cosine similarity to the query.

        FAISS stores L2-normalized vectors via HuggingFaceEmbeddings defaults? No -
        we explicitly normalize here so dot-product == cosine.
        """
        if not docs:
            return []
        q_vec = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec) + 1e-12
        d_vecs = np.array(self.embeddings.embed_documents([d.page_content for d in docs]), dtype=np.float32)
        d_vecs /= np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-12
        sims = d_vecs @ q_vec
        return list(zip(docs, sims.tolist()))


def build_retriever() -> HybridRetriever:
    docs = load_documents()

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = TOP_K

    ensemble = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT],
    )

    return HybridRetriever(
        ensemble=ensemble,
        vectorstore=vectorstore,
        embeddings=embeddings,
        documents=docs,
    )


if __name__ == "__main__":
    r = build_retriever()
    for q in [
        "how do I reset my password",
        "my 2fa codes stopped working after I got a new phone",
        "what's the largest file I can upload through the website",
    ]:
        print(f"\nQuery: {q}")
        results = r.retrieve(q)
        scored = r.score_query(q, results)
        for doc, sim in scored:
            print(f"  [{sim:.3f}] {doc.metadata['source']} :: {doc.metadata['title']}")
