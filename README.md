📄 Resume Finder — RAG-Based Resume Search Engine

Resume Finder is a Retrieval-Augmented Generation (RAG) system that enables semantic search over resumes using natural language job descriptions. It allows recruiters or hiring managers to paste a job description and instantly retrieve the most relevant candidate resumes, ranked by semantic relevance and contextual similarity.

🚀 Features

🔍 Natural Language Search
Input a job description and retrieve resumes ranked by semantic similarity

🧠 RAG Pipeline
Uses embeddings + vector search + reranking to improve relevance

📊 Two-Stage Retrieval

Dense vector similarity search (ChromaDB)

Cross-encoder reranking (FlashRank / MS MARCO model)

⚡ Fast + Scalable

Persistent vector store

Efficient embedding generation using SentenceTransformers

🧾 Multi-Resume Ranking
Aggregates passage-level matches into a resume-level leaderboard using:

max relevance score

hit frequency

best-matching section

🏗️ System Architecture
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
🧩 Tech Stack
Component	Technology
Embeddings	BAAI/bge-large-en-v1.5 (SentenceTransformers)
Vector DB	ChromaDB (persistent local store)
Reranker	FlashRank (ms-marco-MiniLM-L-12-v2)
Language	Python
Storage	Local disk persistence
Pipeline Type	Retrieval-Augmented Generation (RAG)
📂 Project Structure
resume-finder/
│
├── rag_retriever.py        # Main retrieval + ranking logic
├── pdf_loader.ipynb        # Builds vector DB from resume PDFs
├── data/
│   └── vector_store/       # Persisted ChromaDB collection
├── requirements.txt
└── README.md
⚙️ Setup Instructions
1️⃣ Install dependencies
pip install -r requirements.txt

Key libraries used:

sentence-transformers

chromadb

flashrank

2️⃣ Build the Vector Store (one-time)

Run the notebook:

pdf_loader.ipynb

This:

parses resume PDFs

splits them into chunks

generates embeddings

stores them in ChromaDB

3️⃣ Run a Search
from rag_retriever import search

job_description = """
Looking for a backend engineer with experience in Python, distributed systems,
APIs, and cloud infrastructure.
"""

results = search(job_description)

for filename, stats in results[:5]:
    print(filename, stats["max_score"])
📊 Output Format

Each result contains:

[
  (
    "resume_John_Doe.pdf",
    {
      "max_score": 0.92,
      "hit_count": 4,
      "top_text": "Most relevant resume section..."
    }
  ),
  ...
]
🧠 How It Works
Step 1 — Embedding Generation

Converts job description into dense vector representation

Step 2 — Vector Retrieval

Searches resume chunks in ChromaDB using cosine similarity

Step 3 — Reranking

Uses cross-encoder model to improve ranking quality

Step 4 — Resume Aggregation

Combines chunk-level scores into resume-level ranking
