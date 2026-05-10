"""Project 5: RAG game recommendation system."""

from .data import GameDocument, clean_text, load_game_documents
from .embeddings import DEFAULT_EMBEDL_MODEL, EmbedlEmbeddings
from .rag import build_rag_chain, build_recommendation_prompt, format_retrieved_games

__all__ = [
    "GameDocument",
    "clean_text",
    "load_game_documents",
    "DEFAULT_EMBEDL_MODEL",
    "EmbedlEmbeddings",
    "build_rag_chain",
    "build_recommendation_prompt",
    "format_retrieved_games",
]
