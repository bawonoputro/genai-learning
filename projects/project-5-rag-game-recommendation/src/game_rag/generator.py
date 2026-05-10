from __future__ import annotations


def build_local_llm(model_name: str = "distilgpt2", max_new_tokens: int = 160):
    """Create a LangChain-wrapped Hugging Face text generation pipeline."""

    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers and langchain-huggingface are required for generation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    generator = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256,
    )
    return HuggingFacePipeline(pipeline=generator)


def invoke_llm(llm, prompt: str) -> str:
    """Invoke a LangChain LLM across old and new APIs."""

    if hasattr(llm, "invoke"):
        return str(llm.invoke(prompt))
    return str(llm(prompt))
