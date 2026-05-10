import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_rag.data import GameDocument, clean_text, load_game_documents
from game_rag.embeddings import DEFAULT_EMBEDL_MODEL, EmbedlEmbeddings
from game_rag.rag import build_rag_chain, build_recommendation_prompt, format_retrieved_games


class GameRagCoreTests(unittest.TestCase):
    def test_clean_text_removes_html_nulls_and_extra_spaces(self):
        raw = "<p>Explore&nbsp;space!</p>\n\nNull here: nan     with   spaces"

        self.assertEqual(clean_text(raw), "Explore space! Null here: with spaces")

    def test_load_game_documents_combines_title_and_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            desc_csv = tmp_path / "steam_description_data.csv"
            games_csv = tmp_path / "steam.csv"

            with desc_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["steam_appid", "about_the_game", "detailed_description"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "steam_appid": "10",
                        "about_the_game": "<b>Fast co-op shooter</b>",
                        "detailed_description": "Play with friends online.",
                    }
                )
                writer.writerow(
                    {
                        "steam_appid": "20",
                        "about_the_game": "",
                        "detailed_description": "   ",
                    }
                )

            with games_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["appid", "name"])
                writer.writeheader()
                writer.writerow({"appid": "10", "name": "Co-op Blaster"})
                writer.writerow({"appid": "20", "name": "Empty Game"})

            docs = load_game_documents(desc_csv, games_csv)

        self.assertEqual(len(docs), 1)
        self.assertEqual(
            docs[0],
            GameDocument(
                title="Co-op Blaster",
                steam_appid="10",
                description="Fast co-op shooter Play with friends online.",
            ),
        )
        self.assertIn("Title: Co-op Blaster", docs[0].page_content)

    def test_format_retrieved_games_numbers_titles_and_scores(self):
        docs = [
            GameDocument("Story RPG", "1", "A story rich role-playing game.", 0.91),
            GameDocument("Puzzle Lake", "2", "A relaxing puzzle game.", None),
        ]

        formatted = format_retrieved_games(docs)

        self.assertIn("1. Story RPG (score: 0.910)", formatted)
        self.assertIn("A story rich role-playing game.", formatted)
        self.assertIn("2. Puzzle Lake", formatted)

    def test_build_recommendation_prompt_includes_role_query_and_context(self):
        context = "1. Story RPG - narrative fantasy adventure"

        prompt = build_recommendation_prompt("I want a story game", context)

        self.assertIn("You are a game recommender", prompt)
        self.assertIn("I want a story game", prompt)
        self.assertIn(context, prompt)
        self.assertIn("recommend 3 games", prompt.lower())

    def test_embedl_embeddings_default_to_real_embedl_huggingface_model(self):
        embeddings = EmbedlEmbeddings()

        self.assertEqual(DEFAULT_EMBEDL_MODEL, "embedl/all-MiniLM-L6-v2-quantized-trt")
        self.assertEqual(embeddings.model_name, DEFAULT_EMBEDL_MODEL)
        self.assertTrue(embeddings.uses_embedl_model)

    def test_build_rag_chain_returns_langchain_runnable_when_available(self):
        class FakeRetriever:
            def invoke(self, query):
                return [GameDocument("Story RPG", "1", "A story rich adventure.")]

        class FakeLLM:
            def invoke(self, prompt):
                return f"LLM saw: {prompt[:30]}"

        chain = build_rag_chain(FakeRetriever(), FakeLLM())

        self.assertTrue(hasattr(chain, "invoke"))
        response = chain.invoke("I want a story game")
        self.assertIn("LLM saw:", response)


if __name__ == "__main__":
    unittest.main()
