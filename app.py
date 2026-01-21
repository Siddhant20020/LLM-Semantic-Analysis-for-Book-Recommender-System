import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer



# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Semantic Based Book Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

.search-card {
    background: #0d1117;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #30363d;
}

.book-card {
    background: #161b22;
    border-radius: 14px;
    padding: 12px;
    height: 440px;
    display: grid;
    grid-template-rows: 200px 48px 18px auto;
    gap: 8px;
    transition: transform 0.2s ease;
}

.book-card:hover {
    transform: translateY(-4px);
}

.book-img {
    display: flex;
    justify-content: center;
    align-items: center;
    background: #0d1117;
    border-radius: 10px;
}

.book-img img {
    max-height: 180px;
    max-width: 100%;
    object-fit: contain;
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

.emotion-box {
    background: #0d1117;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 6px;
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
    "Joy 😊": "joy",
    "Sadness 😢": "sadness",
    "Anger 😡": "anger",
    "Fear 😱": "fear",
    "Surprise 😮": "surprise",
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
# HYBRID RECOMMENDER (FIXED)
# =====================================================
def recommend(query, category, emotion_col):
    qemb = model.encode([query])
    res = collection.query(query_embeddings=qemb, n_results=60)

    ids = res["ids"][0]
    df = books_df[books_df["isbn13"].isin(ids)].copy()

    # Preserve semantic rank
    df["semantic_score"] = df["isbn13"].apply(lambda x: 1 - ids.index(x) / len(ids))

    if emotion_col:
        df["emotion_score"] = df[emotion_col]
    else:
        df["emotion_score"] = 0.0

    df["final_score"] = 0.75 * df["semantic_score"] + 0.25 * df["emotion_score"]

    if category != "All":
        df = df[df["simple_categories"] == category]

    return df.sort_values("final_score", ascending=False).reset_index(drop=True)

def similar_books(desc, isbn):
    emb = model.encode([desc])
    res = collection.query(query_embeddings=emb, n_results=8)
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
            <img src="{book['large_thumbnail']}">
        </div>
        <div class="book-title">{book['title']}</div>
        <div class="book-author">{book['authors']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.button(
        "View Details",
        key=f"view_{book['isbn13']}",
        use_container_width=True,
        on_click=lambda: st.session_state.update(
            page="detail",
            current_book=book["isbn13"]
        )
    )

# =====================================================
# HOME
# =====================================================
def home():
    st.title("📚 AI Book Recommendation")

    with st.sidebar:
    

        query = st.text_input("Describe a book")
        category = st.selectbox(
            "Category",
            ["All"] + sorted(books_df["simple_categories"].dropna().unique())
        )
        emotion_label = st.selectbox(
            "Emotion",
            ["All"] + list(EMOTION_MAP.keys())
        )

        search = st.button("🔍 Search", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if search and query:
        emotion_col = EMOTION_MAP.get(emotion_label)
        st.session_state.results = recommend(query, category, emotion_col)

    if "results" not in st.session_state:
        return

    results = st.session_state.results
    cols = st.columns(4)

    for i, (_, book) in enumerate(results.head(PAGE_SIZE).iterrows()):
        with cols[i % 4]:
            book_card(book)

# =====================================================
# DETAIL PAGE
# =====================================================
def detail():
    book = books_df[books_df["isbn13"] == st.session_state.current_book].iloc[0]

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.header(book["title"])
    st.image(book["large_thumbnail"], width=260)
    st.write(book["description"])

    st.subheader("🧠 Emotional Profile")
    for label, col in EMOTION_MAP.items():
        st.markdown(f"<div class='emotion-box'><strong>{label}</strong></div>", unsafe_allow_html=True)
        st.progress(float(book[col]))

    st.subheader("📚 Similar Books")
    cols = st.columns(4)
    for i, (_, sim) in enumerate(similar_books(book["description"], book["isbn13"]).iterrows()):
        with cols[i % 4]:
            book_card(sim)
            


# =====================================================
# ROUTER
# =====================================================
if st.session_state.page == "home":
    home()
else:
    detail()
