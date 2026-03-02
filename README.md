# 📄 Resume Finder — RAG-Based Resume Search Engine

**Resume Finder** is a Retrieval-Augmented Generation (RAG) system that enables **semantic search over resumes using natural language job descriptions**. It allows recruiters or hiring managers to paste a job description and instantly retrieve the **most relevant candidate resumes**, ranked by semantic relevance and contextual similarity.

---

## 🚀 Features

- 🔍 **Natural Language Search**  
  Input a job description and retrieve resumes ranked by semantic similarity

- 🧠 **RAG Pipeline**  
  Uses embeddings + vector search + reranking to improve relevance

- 📊 **Two-Stage Retrieval**
  - Dense vector similarity search (ChromaDB)
  - Cross-encoder reranking (FlashRank / MS MARCO model)

- ⚡ **Fast + Scalable**
  - Persistent vector store
  - Efficient embedding generation using SentenceTransformers

- 🧾 **Multi-Resume Ranking**
  Aggregates passage-level matches into a resume-level leaderboard using:
  - max relevance score
  - hit frequency
  - best-matching section

---

## 🏗️ System Architecture
            ┌──────────────────────────────┐
            │      Job Description         │
            └─────────────┬────────────────┘
                          │
                Embedding Generation
      (BAAI/bge-large-en-v1.5 via SentenceTransformers)
                          │
                          ▼
           ┌──────────────────────────────┐
           │     Vector Store (ChromaDB)  │
           │  Precomputed Resume Chunks   │
           └─────────────┬────────────────┘
                          │
                   Top-K Retrieval
                          │
                          ▼
           ┌──────────────────────────────┐
           │      FlashRank Reranker      │
           │  (MS MARCO MiniLM cross-enc) │
           └─────────────┬────────────────┘
                          │
                          ▼
             Resume-Level Aggregation
                          │
                          ▼
                 Ranked Resume Results

              
---

## 🧩 Tech Stack

| Component | Technology |
|----------|------------|
| Embeddings | `BAAI/bge-large-en-v1.5` (SentenceTransformers) |
| Vector DB | ChromaDB (persistent local store) |
| Reranker | FlashRank (`ms-marco-MiniLM-L-12-v2`) |
| Language | Python |
| Storage | Local disk persistence |
| Pipeline Type | Retrieval-Augmented Generation (RAG) |

---

## 📂 Project Structure
resume-finder/
│
├── rag_retriever.py # Main retrieval + ranking logic
├── pdf_loader.ipynb # Builds vector DB from resume PDFs
├── data/
│ └── vector_store/ # Persisted ChromaDB collection
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
