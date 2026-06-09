import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── Paths ──────────────────────────────────────────────────
DOCS_PATH    = "docs"
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"


def load_pdfs(docs_path):
    """Load all PDF files from the given directory and return a list of documents."""
    all_documents = []
    for filename in sorted(os.listdir(docs_path)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_path, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"   ✅ Loaded: {filename} ({len(docs)} pages)")
    return all_documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def embed_and_store(chunks, embed_model=EMBED_MODEL, chroma_path=CHROMA_PATH):
    """Embed document chunks and store them in ChromaDB."""
    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    return vectorstore


def main():
    print("📄 Loading PDFs from docs folder...")

    # ── Step 1: Load all PDFs ──────────────────────────────────
    all_documents = load_pdfs(DOCS_PATH)
    print(f"\n📚 Total pages loaded: {len(all_documents)}")

    # ── Step 2: Split into chunks ──────────────────────────────
    print("\n✂️  Splitting into chunks...")
    chunks = split_documents(all_documents)
    print(f"   ✅ Total chunks created: {len(chunks)}")

    # ── Step 3: Embed and store in ChromaDB ───────────────────
    print("\n🔢 Embedding and storing in ChromaDB...")
    print("   (This may take 2-3 minutes — Ollama is processing...)")
    embed_and_store(chunks)

    print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB.")
    print("   Your knowledge base is ready!")


if __name__ == "__main__":
    main()
