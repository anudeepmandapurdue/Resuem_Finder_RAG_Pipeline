"""
Resume Finder - Job description search using the RAG pipeline.
Uses the pre-built vector store from pdf_loader.ipynb (run that notebook once to populate).
"""
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from flashrank import Ranker, RerankRequest


class EmbeddingManager:
    """Handles query embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: list[str]):
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    """Thin wrapper over ChromaDB for loading the pre-built collection."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str | None = None,
    ):
        base = Path(__file__).resolve().parent
        self.persist_directory = persist_directory or str(base / "data" / "vector_store")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )


class RagRetriever:
    """Retrieves and ranks resumes by job description match."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")

    def find_best_document(self, job_description: str, fetch_k: int = 25) -> list:
        query = job_description.strip()
        query_embeddings = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=fetch_k,
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        passages = [
            {"id": doc_id, "text": text, "meta": meta}
            for doc_id, text, meta in zip(ids, documents, metadatas)
        ]

        try:
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = self.ranker.rerank(rerank_request)
        except (ValueError, Exception):
            # Fallback if FlashRank has compatibility issues: use vector store order
            reranked_results = [
                {"id": iid, "text": d, "meta": m, "score": 1.0 - (i * 0.02)}
                for i, (iid, d, m) in enumerate(zip(ids, documents, metadatas))
            ]

        leaderboard = {}
        for res in reranked_results:
            filename = res["meta"].get("file_name", "Unknown File")
            score = res["score"]

            if filename not in leaderboard:
                leaderboard[filename] = {
                    "max_score": 0.0,
                    "hit_count": 0,
                    "top_text": res["text"],
                }

            if score > leaderboard[filename]["max_score"]:
                leaderboard[filename]["max_score"] = score
                leaderboard[filename]["top_text"] = res["text"]

            leaderboard[filename]["hit_count"] += 1

        return sorted(leaderboard.items(), key=lambda x: x[1]["max_score"], reverse=True)


def get_retriever() -> RagRetriever:
    """Initialize and return the RAG retriever (loads models + vector store)."""
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    return RagRetriever(vector_store, embedding_manager)


def search(job_description: str, fetch_k: int = 20) -> list:
    """
    Run resume search for the given job description.
    Returns list of (filename, stats) sorted by match score.
    """
    retriever = get_retriever()
    return retriever.find_best_document(job_description, fetch_k=fetch_k)
