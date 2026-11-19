"""Movie recommendation RAG pipeline.

Reusable version of the pipeline in ``MovieRAG.ipynb``:
movies.json -> LangChain Documents -> MiniLM embeddings -> Chroma -> retriever
-> Qwen2.5-0.5B-Instruct answer grounded in the retrieved movies.

Run ``python movie_rag.py`` for the terminal chatbot, or ``streamlit run app.py``
for the web demo.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "movies.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COLLECTION_NAME = "movies_collection"
NOT_FOUND_MESSAGE = "I don't have that information in the catalog."

PROMPT = PromptTemplate(
    template="""
You are a knowledgeable movie expert assistant.

Use ONLY the movie catalog context below to answer the question.

If the user asks about movies in a genre (e.g., "what sci-fi movies do you have"),
provide a comprehensive list of all available movies in that genre with their ratings and descriptions.

If the user asks about a specific movie, provide detailed information about that movie.

When recommending movies, mention the genre, year, and IMDb rating if available.

If the answer cannot be found in the catalog, say:
"I don't have that information in the catalog."

Movie catalog context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


def load_catalog(path: str | Path = CATALOG_PATH) -> dict[str, list[dict]]:
    """Load the ``{genre: [movie, ...]}`` catalog and validate its shape."""
    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict) or not all(isinstance(v, list) for v in catalog.values()):
        raise ValueError(f"{path} must map each genre to a list of movies.")
    return catalog


def catalog_to_documents(catalog: dict[str, list[dict]]) -> list[Document]:
    """Convert each movie into a Document whose text is embedded for search."""
    docs = []
    for genre, movies in catalog.items():
        for movie in movies:
            title = movie.get("title", "")
            year = movie.get("year", "")
            director = movie.get("director", "")
            cast = ", ".join(movie.get("cast", []))
            description = movie.get("description", "")
            rating = movie.get("rating", "")
            duration = movie.get("duration_min", "")
            language = movie.get("language", "")

            content = f"""
Genre: {genre}
Title: {title} ({year})
Director: {director}
Cast: {cast}
Description: {description}
IMDb Rating: {rating}/10
Duration: {duration} minutes
Language: {language}
"""
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "genre": genre,
                        "title": title,
                        "year": year,
                        "director": director,
                        "rating": rating,
                        "language": language,
                        "type": "movie",
                    },
                )
            )
    return docs


def load_embeddings(model_name: str = EMBEDDING_MODEL):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(docs: list[Document], embeddings, persist_directory: str | None = None) -> Chroma:
    """Create the Chroma collection, replacing any previous contents.

    ``persist_directory=None`` keeps the index in memory (used by the web app).
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        # Cosine space gives relevance scores in [0, 1]; ranking matches the
        # notebook because MiniLM embeddings are already L2-normalised.
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.reset_collection()  # avoid duplicate documents on rebuild
    vectorstore.add_documents(docs)
    return vectorstore


def load_llm(model_id: str = LLM_MODEL, max_new_tokens: int = 250, temperature: float = 0.3):
    """Load Qwen as a chat model so its chat template is applied to the prompt."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipeline_llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=pipeline_llm)


def format_context(docs: list[Document]) -> str:
    return "\n".join(doc.page_content.strip() + "\n" for doc in docs)


def generate_answer(llm, question: str, docs: list[Document], **generation_kwargs) -> str:
    """Answer ``question`` using only the retrieved ``docs`` as context.

    ``generation_kwargs`` (e.g. ``max_new_tokens``, ``temperature``) override the
    defaults set in :func:`load_llm` for this call only.
    """
    if not docs:
        return NOT_FOUND_MESSAGE
    message = PROMPT.format(context=format_context(docs), question=question)
    response = llm.invoke(message, pipeline_kwargs=generation_kwargs) if generation_kwargs else llm.invoke(message)
    return getattr(response, "content", response).strip()


def retrieve(vectorstore: Chroma, question: str, k: int = 10, genre: str | None = None):
    """Return ``[(Document, relevance_score), ...]`` best match first."""
    flt = {"genre": genre} if genre else None
    return vectorstore.similarity_search_with_relevance_scores(question, k=k, filter=flt)


def ask(vectorstore: Chroma, llm, question: str, k: int = 10) -> dict:
    """Retrieve + generate. Same output keys as the notebook's RetrievalQA chain."""
    docs = vectorstore.similarity_search(question, k=k)
    return {"result": generate_answer(llm, question, docs), "source_documents": docs}


def main() -> None:
    docs = catalog_to_documents(load_catalog())
    print(f"Indexed {len(docs)} movies.")
    vectorstore = build_vectorstore(docs, load_embeddings(), persist_directory=str(BASE_DIR / "chroma_movies_db"))
    llm = load_llm()

    while True:
        question = input("\nAsk about movies (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = ask(vectorstore, llm, question)
        print("\nDocuments Retrieved:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  {i}. {doc.metadata['title']} ({doc.metadata['genre']}, {doc.metadata['rating']})")
        print("\nBot:", result["result"])


if __name__ == "__main__":
    main()
