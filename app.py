
import time
import streamlit as st
from google import genai
from qdrant_client import QdrantClient

COLLECTION_NAME = "ION_books_v2"
TOP_K = 5
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATE_MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="ION Corpus V2", page_icon="📚", layout="wide")

st.title("ION Corpus V2")
st.caption("Validated corpus • 5 books • 3347 vectors • Qdrant + Gemini")

gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

with st.sidebar:
    st.header("Corpus")
    st.write("Collection:", COLLECTION_NAME)
    st.write("Vector DB: Qdrant")
    st.write("Embeddings:", EMBEDDING_MODEL)
    st.write("Reasoning:", GENERATE_MODEL)
    top_k = st.slider("Number of sources", 3, 10, TOP_K)
    show_sources = st.checkbox("Show retrieved sources", value=True)
question = st.chat_input("Ask the ION Corpus V2...")

def generate_with_retry(prompt, retries=5):
    for attempt in range(retries):
        try:
            return gemini.models.generate_content(
                model=GENERATE_MODEL,
                contents=prompt
            )
        except Exception as e:
            st.warning(f"Generate attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None

if question:
    with st.spinner("Searching corpus..."):
        embed_result = gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question
        )

        query_vector = embed_result.embeddings[0].values

        hits = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )

        context_blocks = []
        for i, point in enumerate(hits.points, start=1):
            payload = point.payload
            context_blocks.append(
                f"[SOURCE {i} | book={payload.get('book')} | chunk={payload.get('chunk_id')}]\n"
                f"{payload.get('text', '')}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are answering using only the provided corpus context.

Question:
{question}

Corpus context:
{context}

Instructions:
- Answer clearly and concisely.
- Use only the context above.
- Mention which source numbers support the answer.
- If the context is insufficient, say so.
"""

        answer = generate_with_retry(prompt)

    st.subheader("Answer")
    if answer is None:
        st.error("Generation unavailable right now. Retrieval succeeded; see sources below.")
    else:
        st.write(answer.text)

    st.subheader("Sources used")
    for i, point in enumerate(hits.points, start=1):
        payload = point.payload
        st.write(f"{i}. book={payload.get('book')} | chunk={payload.get('chunk_id')}")

    if show_sources:
        st.subheader("Retrieved source text")
        for i, point in enumerate(hits.points, start=1):
            payload = point.payload
            with st.expander(f"Source {i}: {payload.get('book')} / chunk {payload.get('chunk_id')}"):
                st.write(payload.get("text", ""))
