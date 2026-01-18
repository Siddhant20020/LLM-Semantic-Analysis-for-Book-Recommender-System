import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AI Book Recommendation", layout="wide")

# =====================================================
# CSS (FIXED HEIGHT + GRID)
# =====================================================
st.markdown("""
<style>
.book-card {
    background: #161b22;
    border-radius: 12px;
    padding: 12px;
    height: 380px;
    display: grid;
    grid-template-rows: 180px 48px 18px auto;
    gap: 6px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.book-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}
.book-img {
    display: flex;
    justify-content: center;
    align-items: center;
}
.book-img img {
    max-height: 180px;
}
.book-title {
    font-size: 14px;
    font-weight: 600;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.book-author {
    font-size: 12px;
    color: #9ca3af;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CONSTANTS
# =====================================================
BOOKS_FILE = "books_with_emotions.csv"
CHROMA_DIR = "chroma_db"
COLLECTION = "books"
FALLBACK_IMAGE = "cover-not-found.jpg"
PAGE_SIZE = 8

EMOTION_MAP = {
    "Happy": "joy",
    "Sad": "sadness",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Surprising": "surprise",
}

# =====================================================
# SESSION
# =====================================================
st.session_state.setdefault("page", "home")
st.session_state.setdefault("current_book", None)
st.session_state.setdefault("page_no", 1)

# =====================================================
# DATA
# =====================================================
@st.cache_data
def load_books():
    df = pd.read_csv(BOOKS_FILE)
    df["isbn13"] = df["isbn13"].astype(str)
    df = df.dropna(subset=["description"])
    df["thumbnail"] = df["thumbnail"].fillna("")
    df["large_thumbnail"] = np.where(
        df["thumbnail"] == "",
        FALLBACK_IMAGE,
        df["thumbnail"] + "&fife=w400"
    )
    return df

books_df = load_books()

# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# CHROMA
# =====================================================
@st.cache_resource
def init_chroma():
    return chromadb.Client(Settings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    ))

client = init_chroma()
collection = client.get_or_create_collection(
    COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

@st.cache_resource
def populate_once():
    existing = set(collection.get(include=[])["ids"])
    docs, ids = [], []

    for _, r in books_df.iterrows():
        if r["isbn13"] not in existing:
            docs.append(r["description"])
            ids.append(r["isbn13"])

    if docs:
        emb = model.encode(docs, batch_size=32, show_progress_bar=True)
        collection.add(documents=docs, embeddings=emb, ids=ids)

populate_once()

# =====================================================
# RECOMMEND
# =====================================================
def recommend(query, category, emotion):
    qemb = model.encode([query])
    res = collection.query(query_embeddings=qemb, n_results=50)
    df = books_df[books_df["isbn13"].isin(res["ids"][0])]

    if category != "All":
        df = df[df["simple_categories"] == category]
    if emotion != "All":
        df = df.sort_values(EMOTION_MAP[emotion], ascending=False)

    return df.reset_index(drop=True)

def similar_books(desc, isbn):
    emb = model.encode([desc])
    res = collection.query(query_embeddings=emb, n_results=7)
    return books_df[
        (books_df["isbn13"].isin(res["ids"][0])) &
        (books_df["isbn13"] != isbn)
    ]

# =====================================================
# COMPONENT
# =====================================================
def book_card(book):
    st.markdown(f"""
    <div class="book-card">
        <div class="book-img">
            <img src="{book['large_thumbnail']}" loading="lazy">
        </div>
        <div class="book-title">{book['title']}</div>
        <div class="book-author">{book['authors']}</div>
        <div></div>
    </div>
    """, unsafe_allow_html=True)

    st.button(
        "View Details",
        key=f"view_{book['isbn13']}",
        on_click=lambda: (
            st.session_state.update(
                page="detail",
                current_book=book["isbn13"]
            )
        )
    )

# =====================================================
# HOME
# =====================================================
def home():
    st.title("📚 AI Book Recommendation")

    with st.sidebar:
        category = st.selectbox(
            "Category",
            ["All"] + sorted(books_df["simple_categories"].dropna().unique())
        )
        emotion = st.selectbox(
            "Emotion",
            ["All"] + list(EMOTION_MAP.keys())
        )
        query = st.text_input("Describe a book")

    if st.button("Search") and query:
        st.session_state.results = recommend(query, category, emotion)
        st.session_state.page_no = 1

    if "results" not in st.session_state:
        return

    results = st.session_state.results
    page = st.session_state.page_no

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    cols = st.columns(4)
    for i, (_, book) in enumerate(results.iloc[start:end].iterrows()):
        with cols[i % 4]:
            book_card(book)

    c1, _, c2 = st.columns([1, 6, 1])
    with c1:
        if page > 1 and st.button("⬅ Prev"):
            st.session_state.page_no -= 1
            st.rerun()
    with c2:
        if end < len(results) and st.button("Next ➡"):
            st.session_state.page_no += 1
            st.rerun()

# =====================================================
# DETAIL
# =====================================================
def detail():
    book = books_df[books_df["isbn13"] == st.session_state.current_book].iloc[0]

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.header(book["title"])
    st.image(book["large_thumbnail"], width=260)
    st.write(book["description"])

    st.subheader("📚 Similar Books")
    sims = similar_books(book["description"], book["isbn13"])

    cols = st.columns(4)
    for i, (_, sim) in enumerate(sims.iterrows()):
        with cols[i % 4]:
            book_card(sim)

# =====================================================
# ROUTER
# =====================================================
if st.session_state.page == "home":
    home()
else:
    detail()
    