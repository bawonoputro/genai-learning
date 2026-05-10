from __future__ import annotations

from typing import Any

DEFAULT_EMBEDL_MODEL = "embedl/all-MiniLM-L6-v2-quantized-trt"
DEFAULT_EMBEDL_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDL_PT2_FILE = "embedl_all-MiniLM-L6-v2_int8.pt2"


class EmbedlEmbeddings:
    """LangChain-compatible embedding adapter using an Embedl HF model by default.

    The default model is Embedl's quantized Hugging Face artifact:
    `embedl/all-MiniLM-L6-v2-quantized-trt`.

    That model is public on Hugging Face and distributed as optimized PT2/ONNX
    artifacts. This class therefore supports two execution paths:

    1. `backend="embedl_pt2"` (default for Embedl model ids): download and run
       the Embedl `.pt2` artifact with `torch.export.load`.
    2. `backend="sentence_transformers"`: useful for local fallback experiments
       with standard SentenceTransformer models.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDL_MODEL,
        device: str | None = None,
        backend: str | None = None,
        base_model_name: str = DEFAULT_EMBEDL_BASE_MODEL,
        pt2_filename: str = DEFAULT_EMBEDL_PT2_FILE,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.backend = backend or (
            "embedl_pt2" if model_name.startswith("embedl/") else "sentence_transformers"
        )
        self.base_model_name = base_model_name
        self.pt2_filename = pt2_filename
        self.model_kwargs = model_kwargs
        self._model = None
        self._tokenizer = None

    @property
    def uses_embedl_model(self) -> bool:
        return self.model_name.startswith("embedl/")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for Embedl tokenization. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is not None:
            return self._model

        if self.backend == "embedl_pt2":
            self._model = self._load_embedl_pt2_model()
        elif self.backend == "sentence_transformers":
            self._model = self._load_sentence_transformer_model()
        else:
            raise ValueError(
                f"Unsupported embedding backend '{self.backend}'. "
                "Use 'embedl_pt2' or 'sentence_transformers'."
            )
        return self._model

    def _load_embedl_pt2_model(self):
        try:
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "torch and huggingface_hub are required for Embedl PT2 embeddings. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            artifact_path = hf_hub_download(
                repo_id=self.model_name,
                filename=self.pt2_filename,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download the Embedl embedding model from Hugging Face. "
                "The model is public, so login is usually not required, but "
                "`huggingface-cli login` can help if you hit rate limits. "
                "For a non-Embedl fallback experiment, pass "
                "--embedding-model sentence-transformers/all-MiniLM-L6-v2."
            ) from exc

        # The ExportedProgram captured the model in eval mode at export time;
        # the .module() wrapper does not support .eval() / train-mode toggling.
        model = torch.export.load(artifact_path).module()
        if self.device:
            model.to(self.device)
        return model

    def _load_sentence_transformer_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for this embedding backend. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs = dict(self.model_kwargs)
        if self.device:
            kwargs["device"] = self.device
        return SentenceTransformer(self.model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "sentence_transformers":
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=True
            )
            return embeddings.tolist()
        return [self._embed_one_with_embedl(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.backend == "sentence_transformers":
            embedding = self.model.encode(
                [text], convert_to_numpy=True, show_progress_bar=False
            )[0]
            return embedding.tolist()
        return self._embed_one_with_embedl(text)

    def __call__(self, text: str) -> list[float]:
        """Support LangChain FAISS versions that still call embeddings directly."""

        return self.embed_query(text)

    def _embed_one_with_embedl(self, text: str) -> list[float]:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("torch is required for Embedl PT2 embeddings.") from exc

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        if self.device:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            embedding = self.model(encoded["input_ids"], encoded["attention_mask"])

        normalized = F.normalize(embedding, p=2, dim=1)
        return normalized.squeeze(0).detach().cpu().tolist()
