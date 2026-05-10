from __future__ import annotations

from pathlib import Path

from .data import GameDocument, load_game_documents, to_langchain_documents
from .embeddings import DEFAULT_EMBEDL_MODEL, EmbedlEmbeddings


def create_faiss_index(
    descriptions_csv: Path | str,
    games_csv: Path | str | None,
    index_dir: Path | str,
    *,
    limit: int | None = None,
    embedding_model: str = DEFAULT_EMBEDL_MODEL,
):
    """Build and save a FAISS vector index from Steam game descriptions."""

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "langchain-community and faiss-cpu are required to build the FAISS index. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    game_documents = load_game_documents(descriptions_csv, games_csv, limit=limit)
    langchain_documents = to_langchain_documents(game_documents)
    embeddings = EmbedlEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(langchain_documents, embeddings)
    vector_store.save_local(str(index_dir))
    return vector_store, game_documents


def load_faiss_index(
    index_dir: Path | str,
    *,
    embedding_model: str = DEFAULT_EMBEDL_MODEL,
):
    """Load an existing FAISS vector index from disk."""

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "langchain-community and faiss-cpu are required to load the FAISS index. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    embeddings = EmbedlEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def langchain_docs_to_game_documents(results) -> list[GameDocument]:
    game_docs: list[GameDocument] = []
    for doc in results:
        metadata = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "")
        description = content.split("Description:", 1)[-1].strip() if "Description:" in content else content
        game_docs.append(
            GameDocument(
                title=str(metadata.get("title", "Unknown Game")),
                steam_appid=str(metadata.get("steam_appid", "")),
                description=description,
            )
        )
    return game_docs
