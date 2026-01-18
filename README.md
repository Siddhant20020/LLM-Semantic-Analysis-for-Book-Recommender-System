📚 AI Book Recommendation System

Semantic Search · Emotion-Aware Ranking · Vector Database

🚀 Overview

This project is an LLM-powered semantic book recommendation system that goes beyond traditional keyword-based search by understanding meaning, intent, and emotional context.

Book descriptions and user queries are embedded into a shared vector space using Sentence Transformers, enabling accurate semantic retrieval. Results are further refined using emotion-based ranking, category filtering, and pagination to ensure relevance and scalability.

✨ Key Features

🔍 Semantic Search (LLM Embeddings)

Uses sentence-level embeddings instead of keyword matching

Handles abstract and descriptive user queries


😊 Emotion-Aware Ranking

Re-ranks books using emotion scores (joy, sadness, anger, fear, surprise)

Enables mood-based recommendations


🧠 Persistent Vector Database

Uses ChromaDB for vector storage

Embeddings are generated once and reused across sessions


Lazy-loaded images

Pagination for large result sets

📖 Detail View with Similar Books

Dedicated book detail page

Similar books retrieved via semantic similarity


🏗️ System Architecture
User Query
   ↓
Sentence Embedding Model (MiniLM)
   ↓
Vector Similarity Search (ChromaDB)
   ↓
Candidate Books
   ↓
Emotion + Category Re-Ranking
   ↓
Paginated UI Results



🧰 Tech Stack
Layer	Technology
UI	Streamlit
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
Vector Database	ChromaDB
Data Processing	Pandas, NumPy
Styling	Custom CSS (Grid-based Cards)


🧬 Embedding Strategy

Model: all-MiniLM-L6-v2

Why this model?

Lightweight and fast

Strong semantic performance

Suitable for real-time recommendation systems

Batch Processing

Embeddings generated in batches

Prevents memory spikes during initialization


🗄️ Vector Database Design

💾 Persistent Storage

Embeddings stored on disk (chroma_db/)

No recomputation on application restart


🆔 ID Strategy

ISBN-13 used as unique document ID

Prevents duplicate embeddings


📐 Similarity Metric

Cosine similarity (HNSW index)


📄 Pagination Strategy

Results are fetched once per query

Pagination handled at the UI layer

Prevents repeated vector database queries

Ensures consistent performance with large datasets



🎨 UI Design Decisions

Fixed-height cards for alignment consistency

CSS Grid-based layout

Lazy image loading for performance

Hover animations for better UX

Fallback image for missing thumbnails




▶️ How to Run
1️⃣ Install Dependencies
pip install streamlit pandas numpy chromadb sentence-transformers 
Or,
python install -r requirements.txt

2️⃣ Start the App
streamlit run app.py


⚠️ On first run, embeddings are generated and stored locally.
Subsequent runs reuse existing embeddings automatically.


⚙️ Performance Considerations

Embeddings cached using st.cache_resource

Dataset cached using st.cache_data

Vector search uses HNSW indexing (efficient similarity search)

UI pagination prevents DOM overload



🎯 Project Motivation

This project was built to explore real-world usage of LLM embeddings and vector databases with an emphasis on:

Practical scalability

Semantic understanding

Clean system architecture

Production-style design decisions
