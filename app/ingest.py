import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── Paths ──────────────────────────────────────────────────
DOCS_PATH    = "docs"
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"

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
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)
print(f"   ✅ Total chunks created: {len(chunks)}")

# ── Step 3: Embed and store in ChromaDB ───────────────────
print("\n🔢 Embedding and storing in ChromaDB...")
print("   (This may take 2-3 minutes — Ollama is processing...)")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB.")
print("   Your knowledge base is ready!")