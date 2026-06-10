import streamlit as st
from google import genai
from qdrant_client import QdrantClient

COLLECTION_NAME = "ION_books_v2"
TOP_K = 5
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATE_MODEL = "gemini-2.5-flash"

gemini = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

st.set_page_config(
    page_title="ION Corpus V2",
    page_icon="📚",
    layout="wide"
)

st.title("ION Corpus V2")
st.caption("Validated corpus • 5 books • 3347 vectors")

question = st.text_input(
    "Ask the corpus",
    placeholder="What is ergodicity?"
)

if st.button("Search") and question:

    with st.spinner("Searching corpus..."):

        embed_result = gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question
        )

        query_vector = embed_result.embeddings[0].values

        hits = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K
        )

        context_blocks = []

        for i, point in enumerate(hits.points, start=1):
            payload = point.payload

            context_blocks.append(
                f"[SOURCE {i}]\n"
                f"{payload.get('text', '')}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are answering using only the provided corpus context.

Question:
{question
