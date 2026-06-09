import unittest
from unittest.mock import patch, MagicMock, call
from langchain_core.documents import Document

from app.rag_pipeline import (
    format_docs,
    build_rag_chain,
    run_interactive_loop,
    PROMPT_TEMPLATE,
)


class TestFormatDocs(unittest.TestCase):
    """Tests for the format_docs helper."""

    def test_single_document(self):
        docs = [Document(page_content="Hello world")]
        self.assertEqual(format_docs(docs), "Hello world")

    def test_multiple_documents(self):
        docs = [
            Document(page_content="First"),
            Document(page_content="Second"),
            Document(page_content="Third"),
        ]
        self.assertEqual(format_docs(docs), "First\n\nSecond\n\nThird")

    def test_empty_list(self):
        self.assertEqual(format_docs([]), "")

    def test_documents_with_whitespace(self):
        docs = [
            Document(page_content="  spaces  "),
            Document(page_content="\nnewline\n"),
        ]
        result = format_docs(docs)
        self.assertIn("  spaces  ", result)
        self.assertIn("\nnewline\n", result)
        self.assertIn("\n\n", result)

    def test_documents_with_special_characters(self):
        docs = [Document(page_content="Price: $100 @ 5% tax")]
        self.assertEqual(format_docs(docs), "Price: $100 @ 5% tax")


class TestPromptTemplate(unittest.TestCase):
    """Verify the prompt template contains required placeholders."""

    def test_contains_context_placeholder(self):
        self.assertIn("{context}", PROMPT_TEMPLATE)

    def test_contains_question_placeholder(self):
        self.assertIn("{question}", PROMPT_TEMPLATE)

    def test_contains_company_name(self):
        self.assertIn("Deccan Power & Gas Utilities Ltd", PROMPT_TEMPLATE)

    def test_contains_helpline_fallback(self):
        self.assertIn("1800-XXX-XXXX", PROMPT_TEMPLATE)


class TestBuildRagChain(unittest.TestCase):
    """Tests for build_rag_chain construction."""

    @patch("app.rag_pipeline.StrOutputParser")
    @patch("app.rag_pipeline.PromptTemplate")
    @patch("app.rag_pipeline.OllamaLLM")
    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.OllamaEmbeddings")
    def test_builds_chain_with_defaults(
        self, mock_embed, mock_chroma, mock_llm, mock_prompt, mock_parser
    ):
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs
        mock_vs.as_retriever.return_value = MagicMock()

        build_rag_chain()

        mock_embed.assert_called_once_with(model="nomic-embed-text")
        mock_chroma.assert_called_once()
        mock_llm.assert_called_once_with(model="llama3")
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})

    @patch("app.rag_pipeline.StrOutputParser")
    @patch("app.rag_pipeline.PromptTemplate")
    @patch("app.rag_pipeline.OllamaLLM")
    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.OllamaEmbeddings")
    def test_builds_chain_with_custom_models(
        self, mock_embed, mock_chroma, mock_llm, mock_prompt, mock_parser
    ):
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs
        mock_vs.as_retriever.return_value = MagicMock()

        build_rag_chain(
            chroma_path="/custom/db",
            embed_model="custom-embed",
            llm_model="custom-llm",
        )

        mock_embed.assert_called_once_with(model="custom-embed")
        mock_llm.assert_called_once_with(model="custom-llm")
        chroma_kwargs = mock_chroma.call_args[1]
        self.assertEqual(chroma_kwargs["persist_directory"], "/custom/db")


class TestRunInteractiveLoop(unittest.TestCase):
    """Tests for the interactive Q&A loop."""

    @patch("builtins.input", return_value="exit")
    def test_exit_immediately(self, mock_input):
        chain = MagicMock()
        run_interactive_loop(chain)
        chain.invoke.assert_not_called()

    @patch("builtins.input", side_effect=["What is the tariff?", "exit"])
    def test_asks_one_question_then_exits(self, mock_input):
        chain = MagicMock()
        chain.invoke.return_value = "The tariff is $10/kWh."

        run_interactive_loop(chain)

        chain.invoke.assert_called_once_with("What is the tariff?")

    @patch("builtins.input", side_effect=["", "  ", "exit"])
    def test_skips_empty_input(self, mock_input):
        chain = MagicMock()
        run_interactive_loop(chain)
        chain.invoke.assert_not_called()

    @patch("builtins.input", side_effect=["EXIT", "exit"])
    def test_exit_case_insensitive(self, mock_input):
        chain = MagicMock()
        run_interactive_loop(chain)
        chain.invoke.assert_not_called()

    @patch("builtins.input", side_effect=["q1", "q2", "exit"])
    def test_multiple_questions(self, mock_input):
        chain = MagicMock()
        chain.invoke.side_effect = ["answer1", "answer2"]

        run_interactive_loop(chain)

        self.assertEqual(chain.invoke.call_count, 2)
        chain.invoke.assert_any_call("q1")
        chain.invoke.assert_any_call("q2")


if __name__ == "__main__":
    unittest.main()
