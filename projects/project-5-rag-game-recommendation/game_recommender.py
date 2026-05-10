from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from game_rag.generator import build_local_llm, invoke_llm
from game_rag.embeddings import DEFAULT_EMBEDL_MODEL
from game_rag.rag import (
    build_rag_chain,
    format_retrieved_games,
    generate_fallback_answer,
    invoke_retriever,
)
from game_rag.vector_store import (
    create_faiss_index,
    langchain_docs_to_game_documents,
    load_faiss_index,
)

DEFAULT_PROJECT2_DATA = PROJECT_DIR.parent / "project-2-semantic-search" / "data"
DEFAULT_INDEX_DIR = PROJECT_DIR / "vector_index" / "steam_faiss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 5: RAG game recommendations with Embedl embeddings, FAISS, and LangChain."
    )
    parser.add_argument("query", help="Natural-language game request, e.g. 'I want a horror survival game'")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_PROJECT2_DATA)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for quick index builds")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index even if one exists")
    parser.add_argument("--no-llm", action="store_true", help="Skip distilgpt2 generation and print retrieved recommendations")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDL_MODEL)
    parser.add_argument("--llm-model", default="distilgpt2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptions_csv = args.data_dir / "steam_description_data.csv"
    games_csv = args.data_dir / "steam.csv"

    if args.rebuild_index or not args.index_dir.exists():
        print("Building FAISS index from Steam descriptions...")
        vector_store, game_documents = create_faiss_index(
            descriptions_csv,
            games_csv,
            args.index_dir,
            limit=args.limit,
            embedding_model=args.embedding_model,
        )
        print(f"Indexed {len(game_documents)} games at {args.index_dir}")
    else:
        print(f"Loading FAISS index from {args.index_dir}")
        vector_store = load_faiss_index(args.index_dir, embedding_model=args.embedding_model)

    retriever = vector_store.as_retriever(search_kwargs={"k": args.top_k})
    results = invoke_retriever(retriever, args.query)
    retrieved_games = langchain_docs_to_game_documents(results)
    context = format_retrieved_games(retrieved_games)

    print("\nRetrieved context:")
    print(context)

    if args.no_llm:
        print("\nRecommendation:")
        print(generate_fallback_answer(args.query, retrieved_games))
        return

    try:
        llm = build_local_llm(args.llm_model)
        rag_chain = build_rag_chain(retriever, llm)
        answer = invoke_llm(rag_chain, args.query)
    except ImportError as exc:
        print(f"\nLocal LLM unavailable: {exc}")
        answer = generate_fallback_answer(args.query, retrieved_games)

    print("\nRecommendation:")
    print(answer)


if __name__ == "__main__":
    main()
