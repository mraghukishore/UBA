import os
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.ingest import load_pdfs, split_documents, embed_and_store, main


class TestLoadPdfs(unittest.TestCase):
    """Tests for the load_pdfs function."""

    @patch("app.ingest.PyPDFLoader")
    @patch("app.ingest.os.listdir")
    def test_loads_only_pdf_files(self, mock_listdir, mock_loader_cls):
        mock_listdir.return_value = ["bill.pdf", "readme.txt", "tariff.pdf"]
        doc_a = Document(page_content="page a")
        doc_b = Document(page_content="page b")
        loader_instance = MagicMock()
        loader_instance.load.side_effect = [[doc_a], [doc_b]]
        mock_loader_cls.return_value = loader_instance

        result = load_pdfs("/fake/docs")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "page a")
        self.assertEqual(result[1].page_content, "page b")
        self.assertEqual(mock_loader_cls.call_count, 2)

    @patch("app.ingest.PyPDFLoader")
    @patch("app.ingest.os.listdir")
    def test_returns_empty_when_no_pdfs(self, mock_listdir, mock_loader_cls):
        mock_listdir.return_value = ["readme.txt", "data.csv"]

        result = load_pdfs("/fake/docs")

        self.assertEqual(result, [])
        mock_loader_cls.assert_not_called()

    @patch("app.ingest.PyPDFLoader")
    @patch("app.ingest.os.listdir")
    def test_returns_empty_for_empty_directory(self, mock_listdir, mock_loader_cls):
        mock_listdir.return_value = []

        result = load_pdfs("/fake/docs")

        self.assertEqual(result, [])

    @patch("app.ingest.PyPDFLoader")
    @patch("app.ingest.os.listdir")
    def test_multiple_pages_per_pdf(self, mock_listdir, mock_loader_cls):
        mock_listdir.return_value = ["multi.pdf"]
        pages = [Document(page_content=f"page {i}") for i in range(5)]
        loader_instance = MagicMock()
        loader_instance.load.return_value = pages
        mock_loader_cls.return_value = loader_instance

        result = load_pdfs("/fake/docs")

        self.assertEqual(len(result), 5)

    @patch("app.ingest.PyPDFLoader")
    @patch("app.ingest.os.listdir")
    def test_constructs_correct_filepath(self, mock_listdir, mock_loader_cls):
        mock_listdir.return_value = ["report.pdf"]
        loader_instance = MagicMock()
        loader_instance.load.return_value = [Document(page_content="x")]
        mock_loader_cls.return_value = loader_instance

        load_pdfs("/my/docs")

        expected_path = os.path.join("/my/docs", "report.pdf")
        mock_loader_cls.assert_called_once_with(expected_path)


class TestSplitDocuments(unittest.TestCase):
    """Tests for the split_documents function."""

    @patch("app.ingest.RecursiveCharacterTextSplitter")
    def test_uses_default_params(self, mock_splitter_cls):
        docs = [Document(page_content="hello world")]
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = docs
        mock_splitter_cls.return_value = mock_splitter

        result = split_documents(docs)

        mock_splitter_cls.assert_called_once_with(chunk_size=500, chunk_overlap=50)
        mock_splitter.split_documents.assert_called_once_with(docs)
        self.assertEqual(result, docs)

    @patch("app.ingest.RecursiveCharacterTextSplitter")
    def test_uses_custom_params(self, mock_splitter_cls):
        docs = [Document(page_content="text")]
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = ["chunk1", "chunk2"]
        mock_splitter_cls.return_value = mock_splitter

        result = split_documents(docs, chunk_size=200, chunk_overlap=20)

        mock_splitter_cls.assert_called_once_with(chunk_size=200, chunk_overlap=20)
        self.assertEqual(len(result), 2)

    @patch("app.ingest.RecursiveCharacterTextSplitter")
    def test_empty_input_returns_empty(self, mock_splitter_cls):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = []
        mock_splitter_cls.return_value = mock_splitter

        result = split_documents([])

        self.assertEqual(result, [])


class TestEmbedAndStore(unittest.TestCase):
    """Tests for the embed_and_store function."""

    @patch("app.ingest.Chroma")
    @patch("app.ingest.OllamaEmbeddings")
    def test_creates_vectorstore_with_defaults(self, mock_embed_cls, mock_chroma):
        chunks = [Document(page_content="chunk")]
        mock_embeddings = MagicMock()
        mock_embed_cls.return_value = mock_embeddings
        mock_vs = MagicMock()
        mock_chroma.from_documents.return_value = mock_vs

        result = embed_and_store(chunks)

        mock_embed_cls.assert_called_once_with(model="nomic-embed-text")
        mock_chroma.from_documents.assert_called_once_with(
            documents=chunks,
            embedding=mock_embeddings,
            persist_directory="chroma_db"
        )
        self.assertEqual(result, mock_vs)

    @patch("app.ingest.Chroma")
    @patch("app.ingest.OllamaEmbeddings")
    def test_creates_vectorstore_with_custom_params(self, mock_embed_cls, mock_chroma):
        chunks = [Document(page_content="data")]
        mock_chroma.from_documents.return_value = MagicMock()

        embed_and_store(chunks, embed_model="custom-model", chroma_path="/custom/path")

        mock_embed_cls.assert_called_once_with(model="custom-model")
        call_kwargs = mock_chroma.from_documents.call_args[1]
        self.assertEqual(call_kwargs["persist_directory"], "/custom/path")


class TestMain(unittest.TestCase):
    """Tests for the main orchestration function."""

    @patch("app.ingest.embed_and_store")
    @patch("app.ingest.split_documents")
    @patch("app.ingest.load_pdfs")
    def test_main_calls_pipeline_steps(self, mock_load, mock_split, mock_embed):
        mock_load.return_value = [Document(page_content="doc")]
        mock_split.return_value = [Document(page_content="chunk")]

        main()

        mock_load.assert_called_once()
        mock_split.assert_called_once()
        mock_embed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
