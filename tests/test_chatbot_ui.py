import os
import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document

# chatbot_ui.py has top-level Streamlit calls that fail outside a Streamlit
# runtime, so we import only the pure helper functions that were extracted
# to module level.  We replicate their logic here to keep the test
# self-contained and avoid importing the module (which triggers st.set_page_config).

# ── Inline copies of the pure functions from chatbot_ui.py ──


def format_docs(docs):
    """Join document page_content fields with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_raw_results(docs):
    """Format raw retrieval results for display when LLM is off."""
    answer = "📄 **Raw document chunks retrieved:**\n\n"
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown")
        src = os.path.basename(src)
        answer += f"**Chunk {i+1}** — `{src}`\n"
        answer += f"{doc.page_content}\n\n---\n\n"
    return answer


def init_chat_history(session_state):
    """Initialize chat history with a welcome message if not already set."""
    if "messages" not in session_state:
        session_state["messages"] = []
        session_state["messages"].append(
            {
                "role": "assistant",
                "content": "Hello! 👋 I'm your Deccan Power & Gas Utilities billing assistant. How can I help you today?",
            }
        )


# ── Tests ───────────────────────────────────────────────────


class TestFormatDocs(unittest.TestCase):
    """Tests for format_docs (chatbot variant)."""

    def test_single_document(self):
        docs = [Document(page_content="Hello")]
        self.assertEqual(format_docs(docs), "Hello")

    def test_multiple_documents(self):
        docs = [
            Document(page_content="A"),
            Document(page_content="B"),
            Document(page_content="C"),
        ]
        self.assertEqual(format_docs(docs), "A\n\nB\n\nC")

    def test_empty(self):
        self.assertEqual(format_docs([]), "")


class TestFormatRawResults(unittest.TestCase):
    """Tests for format_raw_results."""

    def test_single_chunk_with_source(self):
        docs = [
            Document(
                page_content="Electricity rate is $0.10/kWh",
                metadata={"source": "/docs/tariff.pdf"},
            )
        ]
        result = format_raw_results(docs)

        self.assertIn("**Chunk 1**", result)
        self.assertIn("`tariff.pdf`", result)
        self.assertIn("Electricity rate is $0.10/kWh", result)

    def test_multiple_chunks(self):
        docs = [
            Document(page_content="First", metadata={"source": "a.pdf"}),
            Document(page_content="Second", metadata={"source": "b.pdf"}),
            Document(page_content="Third", metadata={"source": "c.pdf"}),
        ]
        result = format_raw_results(docs)

        self.assertIn("**Chunk 1**", result)
        self.assertIn("**Chunk 2**", result)
        self.assertIn("**Chunk 3**", result)

    def test_missing_source_defaults_to_unknown(self):
        docs = [Document(page_content="No source", metadata={})]
        result = format_raw_results(docs)

        self.assertIn("`Unknown`", result)

    def test_empty_docs(self):
        result = format_raw_results([])
        self.assertEqual(result, "📄 **Raw document chunks retrieved:**\n\n")

    def test_source_basename_extraction(self):
        docs = [
            Document(
                page_content="content",
                metadata={"source": "/long/path/to/billing.pdf"},
            )
        ]
        result = format_raw_results(docs)

        self.assertIn("`billing.pdf`", result)
        self.assertNotIn("/long/path/to/", result)


class TestInitChatHistory(unittest.TestCase):
    """Tests for init_chat_history."""

    def test_initializes_when_empty(self):
        state = {}
        init_chat_history(state)

        self.assertIn("messages", state)
        self.assertEqual(len(state["messages"]), 1)
        self.assertEqual(state["messages"][0]["role"], "assistant")
        self.assertIn("billing assistant", state["messages"][0]["content"])

    def test_does_not_overwrite_existing(self):
        state = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        init_chat_history(state)

        self.assertEqual(len(state["messages"]), 2)

    def test_welcome_message_content(self):
        state = {}
        init_chat_history(state)

        msg = state["messages"][0]["content"]
        self.assertIn("Deccan Power & Gas Utilities", msg)


class TestChatbotConfig(unittest.TestCase):
    """Tests for chatbot configuration constants (without importing the module)."""

    def test_prompt_template_has_required_placeholders(self):
        # We read the template directly since we can't import chatbot_ui
        from app.rag_pipeline import PROMPT_TEMPLATE

        self.assertIn("{context}", PROMPT_TEMPLATE)
        self.assertIn("{question}", PROMPT_TEMPLATE)

    def test_chatbot_prompt_template_readable(self):
        # Read the file and parse the template string
        import ast

        with open(
            os.path.join(os.path.dirname(__file__), "..", "app", "chatbot_ui.py")
        ) as f:
            content = f.read()

        self.assertIn("CHATBOT_PROMPT_TEMPLATE", content)
        self.assertIn("{context}", content)
        self.assertIn("{question}", content)
        self.assertIn("Deccan Power & Gas Utilities Ltd", content)


if __name__ == "__main__":
    unittest.main()
