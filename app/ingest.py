import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── Paths ──────────────────────────────────────────────────
DOCS_PATH    = "docs"
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"


def load_pdfs(docs_path):
    """Load all PDFs from a directory, skipping files that fail to parse."""
    if not os.path.isdir(docs_path):
        raise FileNotFoundError(
            f"Documents directory '{docs_path}' does not exist. "
            "Please create it and add your PDF files."
        )

    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{docs_path}'. "
            "Please add at least one PDF document."
        )

    all_documents = []
    failed_files = []
    for filename in pdf_files:
        filepath = os.path.join(docs_path, filename)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"   ✅ Loaded: {filename} ({len(docs)} pages)")
        except Exception as exc:
            failed_files.append(filename)
            print(f"   ❌ Failed to load '{filename}': {exc}")

    if failed_files:
        print(f"\n⚠️  Skipped {len(failed_files)} file(s) that could not be parsed.")

    if not all_documents:
        raise RuntimeError(
            "No pages were loaded from any PDF. "
            "Check that the files in 'docs/' are valid PDFs."
        )

    return all_documents


def split_documents(documents):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError(
            "Text splitter produced zero chunks. "
            "The loaded PDFs may contain no extractable text."
        )
    return chunks


def embed_and_store(chunks, embed_model, chroma_path):
    """Embed document chunks and store them in ChromaDB."""
    try:
        embeddings = OllamaEmbeddings(model=embed_model)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to initialise Ollama embeddings (model='{embed_model}'). "
            f"Is Ollama running? Error: {exc}"
        ) from exc

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_path
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to store embeddings in ChromaDB at '{chroma_path}': {exc}"
        ) from exc

    return vectorstore


def main():
    print("📄 Loading PDFs from docs folder...")
    try:
        all_documents = load_pdfs(DOCS_PATH)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    print(f"\n📚 Total pages loaded: {len(all_documents)}")

    print("\n✂️  Splitting into chunks...")
    try:
        chunks = split_documents(all_documents)
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    print(f"   ✅ Total chunks created: {len(chunks)}")

    print("\n🔢 Embedding and storing in ChromaDB...")
    print("   (This may take 2-3 minutes — Ollama is processing...)")
    try:
        embed_and_store(chunks, EMBED_MODEL, CHROMA_PATH)
    except (ConnectionError, RuntimeError) as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB.")
    print("   Your knowledge base is ready!")


if __name__ == "__main__":
    main()
