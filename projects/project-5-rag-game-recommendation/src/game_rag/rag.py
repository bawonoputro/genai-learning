from __future__ import annotations

from collections.abc import Iterable

from .data import GameDocument


def format_retrieved_games(documents: Iterable[GameDocument]) -> str:
    """Create a compact context block from retrieved game documents."""

    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        score_text = f" (score: {doc.score:.3f})" if doc.score is not None else ""
        lines.append(f"{index}. {doc.title}{score_text}")
        lines.append(f"   Steam app id: {doc.steam_appid}")
        lines.append(f"   Description: {doc.description[:800]}")
    return "\n".join(lines)


def build_recommendation_prompt(query: str, context: str) -> str:
    """Build the final prompt for the lightweight generator model."""

    return f"""You are a game recommender.
Use the context to recommend 3 games that best match the user's request.
Explain each recommendation in one short sentence.
Only recommend games that appear in the context.

User request:
{query}

Context:
{context}

Answer with:
1. Game title - why it matches
2. Game title - why it matches
3. Game title - why it matches
""".strip()


def docs_from_langchain_results(results) -> list[GameDocument]:
    """Convert LangChain retriever results back into GameDocument values."""

    game_docs: list[GameDocument] = []
    for item in results:
        metadata = getattr(item, "metadata", {}) or {}
        content = getattr(item, "page_content", "")
        description = content.split("Description:", 1)[-1].strip() if "Description:" in content else content
        score = metadata.get("score")
        game_docs.append(
            GameDocument(
                title=str(metadata.get("title", "Unknown Game")),
                steam_appid=str(metadata.get("steam_appid", "")),
                description=description,
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )
    return game_docs


def invoke_retriever(retriever, query: str):
    """Invoke a LangChain retriever across old and new LangChain APIs."""

    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


class SimpleRagChain:
    """Small invoke-compatible fallback used when langchain-core is unavailable."""

    def __init__(self, retriever, llm) -> None:
        self.retriever = retriever
        self.llm = llm

    def invoke(self, query: str) -> str:
        results = invoke_retriever(self.retriever, query)
        game_docs = docs_from_langchain_results(results)
        context = format_retrieved_games(game_docs)
        prompt = build_recommendation_prompt(query, context)
        if hasattr(self.llm, "invoke"):
            return str(self.llm.invoke(prompt))
        return str(self.llm(prompt))


def build_rag_chain(retriever, llm):
    """Build a LangChain LCEL RAG chain from retriever -> prompt -> LLM.

    This satisfies the project goal of using a real LangChain chain while still
    keeping a tiny no-dependency fallback for unit tests and lightweight review.
    """

    try:
        from langchain_core.runnables import Runnable, RunnableLambda
    except ImportError:
        return SimpleRagChain(retriever, llm)

    def retrieve_and_prompt(query: str) -> str:
        results = invoke_retriever(retriever, query)
        game_docs = docs_from_langchain_results(results)
        context = format_retrieved_games(game_docs)
        return build_recommendation_prompt(query, context)

    if isinstance(llm, Runnable) or callable(llm):
        llm_runnable = llm
    else:
        llm_runnable = RunnableLambda(
            lambda prompt: str(llm.invoke(prompt))
            if hasattr(llm, "invoke")
            else str(llm(prompt))
        )

    return RunnableLambda(retrieve_and_prompt) | llm_runnable


def generate_fallback_answer(query: str, retrieved_games: list[GameDocument]) -> str:
    """Deterministic recommendation text when the local LLM is unavailable or disabled."""

    lines = [f"Recommendations for: {query}"]
    for index, game in enumerate(retrieved_games[:3], start=1):
        lines.append(
            f"{index}. {game.title} - This matches because its Steam description is relevant to your request."
        )
    return "\n".join(lines)
