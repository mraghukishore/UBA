import sys
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"


def main():
    print("🔍 Connecting to ChromaDB...")
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    except Exception as exc:
        print(
            f"❌ Failed to connect to ChromaDB at '{CHROMA_PATH}' "
            f"(embed model='{EMBED_MODEL}'): {exc}"
        )
        sys.exit(1)

    try:
        total = vectorstore._collection.count()
    except Exception as exc:
        print(f"❌ Failed to read collection count from ChromaDB: {exc}")
        sys.exit(1)

    print(f"✅ Total chunks in ChromaDB: {total}")
    if total == 0:
        print("⚠️  The database is empty. Run ingest.py first to load documents.")
        sys.exit(1)

    print("\n🔎 Testing a sample query...")
    query = "What is the late payment charge?"
    try:
        results = vectorstore.similarity_search(query, k=3)
    except Exception as exc:
        print(f"❌ Similarity search failed: {exc}")
        sys.exit(1)

    if not results:
        print(f"⚠️  No results returned for query: '{query}'")
        sys.exit(1)

    print(f"\nTop {len(results)} results for: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:200])
        print()

    print("✅ Verification complete — ChromaDB is healthy.")


if __name__ == "__main__":
    main()
