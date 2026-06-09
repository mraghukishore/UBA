import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.verify import (
    connect_to_chroma,
    get_total_chunks,
    query_vectorstore,
    format_results,
    main,
)


class TestConnectToChroma(unittest.TestCase):
    """Tests for connect_to_chroma."""

    @patch("app.verify.Chroma")
    @patch("app.verify.OllamaEmbeddings")
    def test_connects_with_defaults(self, mock_embed, mock_chroma):
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        result = connect_to_chroma()

        mock_embed.assert_called_once_with(model="nomic-embed-text")
        mock_chroma.assert_called_once()
        self.assertEqual(result, mock_vs)

    @patch("app.verify.Chroma")
    @patch("app.verify.OllamaEmbeddings")
    def test_connects_with_custom_params(self, mock_embed, mock_chroma):
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        connect_to_chroma(chroma_path="/custom/path", embed_model="custom-embed")

        mock_embed.assert_called_once_with(model="custom-embed")
        chroma_kwargs = mock_chroma.call_args[1]
        self.assertEqual(chroma_kwargs["persist_directory"], "/custom/path")


class TestGetTotalChunks(unittest.TestCase):
    """Tests for get_total_chunks."""

    def test_returns_count(self):
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 42
        self.assertEqual(get_total_chunks(mock_vs), 42)

    def test_returns_zero_for_empty_store(self):
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 0
        self.assertEqual(get_total_chunks(mock_vs), 0)


class TestQueryVectorstore(unittest.TestCase):
    """Tests for query_vectorstore."""

    def test_returns_search_results(self):
        mock_vs = MagicMock()
        expected = [Document(page_content="result")]
        mock_vs.similarity_search.return_value = expected

        result = query_vectorstore(mock_vs, "test query")

        mock_vs.similarity_search.assert_called_once_with("test query", k=3)
        self.assertEqual(result, expected)

    def test_custom_k_value(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        query_vectorstore(mock_vs, "test", k=5)

        mock_vs.similarity_search.assert_called_once_with("test", k=5)

    def test_returns_empty_for_no_matches(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        result = query_vectorstore(mock_vs, "nonexistent")

        self.assertEqual(result, [])


class TestFormatResults(unittest.TestCase):
    """Tests for format_results."""

    def test_formats_single_result(self):
        docs = [Document(page_content="This is a test document.")]
        result = format_results(docs)

        self.assertIn("--- Chunk 1 ---", result)
        self.assertIn("This is a test document.", result)

    def test_formats_multiple_results(self):
        docs = [
            Document(page_content="First chunk"),
            Document(page_content="Second chunk"),
        ]
        result = format_results(docs)

        self.assertIn("--- Chunk 1 ---", result)
        self.assertIn("--- Chunk 2 ---", result)
        self.assertIn("First chunk", result)
        self.assertIn("Second chunk", result)

    def test_truncates_long_content(self):
        long_text = "A" * 500
        docs = [Document(page_content=long_text)]

        result = format_results(docs, max_chars=100)

        # The displayed content should be truncated to 100 chars
        lines = result.split("\n")
        content_line = lines[1]  # Line after "--- Chunk 1 ---"
        self.assertEqual(len(content_line), 100)

    def test_empty_results(self):
        result = format_results([])
        self.assertEqual(result, "")

    def test_custom_max_chars(self):
        docs = [Document(page_content="Short")]
        result = format_results(docs, max_chars=3)
        self.assertIn("Sho", result)


class TestMain(unittest.TestCase):
    """Tests for the main orchestration function."""

    @patch("app.verify.format_results")
    @patch("app.verify.query_vectorstore")
    @patch("app.verify.get_total_chunks")
    @patch("app.verify.connect_to_chroma")
    def test_main_runs_full_pipeline(
        self, mock_connect, mock_count, mock_query, mock_format
    ):
        mock_vs = MagicMock()
        mock_connect.return_value = mock_vs
        mock_count.return_value = 100
        mock_query.return_value = [Document(page_content="result")]
        mock_format.return_value = "formatted"

        main()

        mock_connect.assert_called_once()
        mock_count.assert_called_once_with(mock_vs)
        mock_query.assert_called_once_with(
            mock_vs, "What is the late payment charge?"
        )
        mock_format.assert_called_once()


if __name__ == "__main__":
    unittest.main()
