from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NULL_WORD_RE = re.compile(r"\b(?:nan|none)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GameDocument:
    """A Steam game document used for retrieval and recommendation."""

    title: str
    steam_appid: str
    description: str
    score: float | None = None

    @property
    def page_content(self) -> str:
        return f"Title: {self.title}\nDescription: {self.description}"

    @property
    def metadata(self) -> dict[str, str]:
        return {"title": self.title, "steam_appid": self.steam_appid}


def clean_text(value: object) -> str:
    """Normalize Steam CSV text for embedding and prompts."""

    if value is None:
        return ""

    text = html.unescape(str(value)).replace("\xa0", " ")
    text = _HTML_TAG_RE.sub(" ", text)
    text = _NULL_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _read_title_lookup(games_csv: Path | str | None) -> dict[str, str]:
    if games_csv is None:
        return {}

    path = Path(games_csv)
    if not path.exists():
        return {}

    lookup: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            appid = clean_text(row.get("appid"))
            name = clean_text(row.get("name"))
            if appid and name:
                lookup[appid] = name
    return lookup


def load_game_documents(
    descriptions_csv: Path | str,
    games_csv: Path | str | None = None,
    *,
    limit: int | None = None,
) -> list[GameDocument]:
    """Load Steam description rows as clean GameDocument objects.

    Args:
        descriptions_csv: Kaggle `steam_description_data.csv` path.
        games_csv: Optional Kaggle `steam.csv` path for appid -> title lookup.
        limit: Optional number of valid documents to return for quick demos.
    """

    descriptions_path = Path(descriptions_csv)
    if not descriptions_path.exists():
        raise FileNotFoundError(f"Steam descriptions CSV not found: {descriptions_path}")

    title_lookup = _read_title_lookup(games_csv)
    documents: list[GameDocument] = []

    with descriptions_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            appid = clean_text(row.get("steam_appid") or row.get("appid"))
            about = clean_text(row.get("about_the_game"))
            detailed = clean_text(row.get("detailed_description"))
            description = clean_text(f"{about} {detailed}")

            if not description:
                continue

            title = title_lookup.get(appid) or clean_text(row.get("name")) or f"Steam app {appid}"
            documents.append(
                GameDocument(title=title, steam_appid=appid, description=description)
            )

            if limit is not None and len(documents) >= limit:
                break

    return documents


def to_langchain_documents(game_documents: Iterable[GameDocument]):
    """Convert GameDocument objects to LangChain Document objects lazily.

    Imported here so the pure data-loading tests do not require LangChain.
    """

    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "LangChain is required for vector-store construction. "
            "Install project requirements first: pip install -r requirements.txt"
        ) from exc

    return [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in game_documents
    ]
