# populate_chroma.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load books
books_df = pd.read_csv("books_with_emotions.csv")
books_df["isbn13"] = books_df["isbn13"].astype(str)
books_df = books_df.dropna(subset=["description"])

# Initialize ChromaDB
client = chromadb.Client(Settings(persist_directory="chroma_db", anonymized_telemetry=False))
collection = client.get_or_create_collection(name="books", metadata={"hnsw:space": "cosine"})

# Check if already populated
if collection.count() == 0:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = books_df["description"].tolist()
    ids = books_df["isbn13"].tolist()
    metadatas = [{"title": row["title"], "category": row["simple_categories"]} for _, row in books_df.iterrows()]
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True)
    
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print("ChromaDB populated successfully!")
else:
    print("ChromaDB already populated. Nothing to do.")
