import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import DOCS_PATH, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from app.shared import get_embeddings

print("📄 Loading PDFs from docs folder...")

# ── Step 1: Load all PDFs ──────────────────────────────────
all_documents = []
for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DOCS_PATH, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"   ✅ Loaded: {filename} ({len(docs)} pages)")

print(f"\n📚 Total pages loaded: {len(all_documents)}")

# ── Step 2: Split into chunks ──────────────────────────────
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(all_documents)
print(f"   ✅ Total chunks created: {len(chunks)}")

# ── Step 3: Embed and store in ChromaDB ───────────────────
print("\n🔢 Embedding and storing in ChromaDB...")
print("   (This may take 2-3 minutes — Ollama is processing...)")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    persist_directory=CHROMA_PATH,
)

print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB.")
print("   Your knowledge base is ready!")
