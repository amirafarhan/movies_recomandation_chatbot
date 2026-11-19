"""Streamlit demo for the Movie Recommendation RAG chatbot.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import movie_rag

MAX_QUESTION_LEN = 500
EXAMPLES = [
    "What sci-fi movies do you have?",
    "Recommend a psychological thriller with a twist ending",
    "Tell me about Inception",
    "Who directed The Shawshank Redemption?",
    "A feel-good animated movie for the family",
]

st.set_page_config(page_title="Movie RAG Chatbot", page_icon="🎬", layout="wide")


@st.cache_data
def get_catalog_df() -> pd.DataFrame:
    catalog = movie_rag.load_catalog()
    rows = [
        {"genre": genre, **movie, "cast": ", ".join(movie.get("cast", []))}
        for genre, movies in catalog.items()
        for movie in movies
    ]
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Building the vector index (first run downloads the embedding model)...")
def get_vectorstore():
    docs = movie_rag.catalog_to_documents(movie_rag.load_catalog())
    return movie_rag.build_vectorstore(docs, movie_rag.load_embeddings())


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B-Instruct (first run downloads ~1 GB)...")
def get_llm():
    # Loaded once and shared; generation settings are passed per request so that
    # changing the sliders does not load another copy of the model.
    return movie_rag.load_llm()


def results_frame(results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "title": d.metadata["title"],
                "genre": d.metadata["genre"],
                "year": d.metadata["year"],
                "rating": d.metadata["rating"],
                "director": d.metadata["director"],
                "relevance": round(score, 3),
            }
            for d, score in results
        ]
    )


def show_sources(df: pd.DataFrame) -> None:
    with st.expander(f"Retrieved movies ({len(df)})", expanded=False):
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                "relevance": st.column_config.ProgressColumn("relevance", min_value=0.0, max_value=1.0, format="%.3f"),
                "year": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.bar_chart(df.set_index("title")["relevance"], horizontal=True, height=40 + 28 * len(df))


df_catalog = get_catalog_df()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_llm = st.toggle("Generate AI answer (Qwen LLM)", value=True,
                        help="Off = semantic search only (fast, no LLM download).")
    k = st.slider("Movies to retrieve (k)", 1, 10, 10, help="Notebook default: 10")
    genre = st.selectbox("Restrict to genre", ["All genres"] + sorted(df_catalog["genre"].unique()))
    with st.expander("LLM options"):
        max_new_tokens = st.slider("Max new tokens", 50, 500, 250, step=25)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.3, step=0.1)
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
    st.caption(f"Embeddings: `{movie_rag.EMBEDDING_MODEL}`  \nLLM: `{movie_rag.LLM_MODEL}`")

st.title("🎬 Movie Recommendation RAG Chatbot")
st.caption("LangChain · ChromaDB · Sentence-Transformers · Qwen2.5 — answers are grounded only in `movies.json`.")

tab_chat, tab_catalog = st.tabs(["💬 Chat", "📚 Catalog explorer"])

# ---------------- Chat tab ----------------
with tab_chat:
    st.session_state.setdefault("messages", [])

    cols = st.columns(len(EXAMPLES))
    example = None
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, width="stretch"):
            example = ex

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") is not None:
                show_sources(msg["sources"])

    question = st.chat_input("Ask about movies in the catalog...") or example

    if question:
        question = question.strip()
        with st.chat_message("user"):
            st.markdown(question)

        if not question:
            st.warning("Please enter a question.")
        elif len(question) > MAX_QUESTION_LEN:
            st.warning(f"Please keep your question under {MAX_QUESTION_LEN} characters.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                try:
                    results = movie_rag.retrieve(
                        get_vectorstore(), question, k=k, genre=None if genre == "All genres" else genre
                    )
                    sources = results_frame(results)
                    if use_llm:
                        with st.spinner("Thinking..."):
                            answer = movie_rag.generate_answer(
                                get_llm(), question, [d for d, _ in results],
                                max_new_tokens=max_new_tokens, temperature=temperature,
                            )
                    elif results:
                        top = sources.iloc[0]
                        answer = (f"**Top match:** {top['title']} ({top['year']}) — {top['genre']}, "
                                  f"IMDb {top['rating']}. See all retrieved movies below.")
                    else:
                        answer = movie_rag.NOT_FOUND_MESSAGE
                except Exception as exc:  # model download / inference failures
                    st.error(f"Something went wrong: {exc}")
                else:
                    st.markdown(answer)
                    show_sources(sources)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ---------------- Catalog tab ----------------
with tab_catalog:
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", len(df_catalog))
    c2.metric("Genres", df_catalog["genre"].nunique())
    c3.metric("Avg. IMDb rating", f"{df_catalog['rating'].mean():.2f}")

    sel = st.multiselect("Genres", sorted(df_catalog["genre"].unique()))
    min_rating = st.slider("Minimum IMDb rating", 0.0, 10.0, 0.0, step=0.1)
    view = df_catalog[df_catalog["rating"] >= min_rating]
    if sel:
        view = view[view["genre"].isin(sel)]

    left, right = st.columns(2)
    with left:
        st.subheader("Average rating by genre")
        st.bar_chart(view.groupby("genre")["rating"].mean().sort_values())
    with right:
        st.subheader("Rating vs. release year")
        st.scatter_chart(view, x="year", y="rating", color="genre")

    st.dataframe(
        view[["title", "genre", "year", "rating", "director", "cast", "duration_min", "language", "description"]]
        .sort_values("rating", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={"year": st.column_config.NumberColumn(format="%d")},
    )
