from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"


def connect_to_chroma(chroma_path=CHROMA_PATH, embed_model=EMBED_MODEL):
    """Connect to ChromaDB and return the vectorstore instance."""
    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    return vectorstore


def get_total_chunks(vectorstore):
    """Return the total number of chunks stored in the vectorstore."""
    return vectorstore._collection.count()


def query_vectorstore(vectorstore, query, k=3):
    """Run a similarity search against the vectorstore."""
    return vectorstore.similarity_search(query, k=k)


def format_results(results, max_chars=200):
    """Format search results for display."""
    lines = []
    for i, doc in enumerate(results):
        lines.append(f"--- Chunk {i+1} ---")
        lines.append(doc.page_content[:max_chars])
        lines.append("")
    return "\n".join(lines)


def main():
    print("🔍 Connecting to ChromaDB...")
    vectorstore = connect_to_chroma()

    total = get_total_chunks(vectorstore)
    print(f"✅ Total chunks in ChromaDB: {total}")

    print("\n🔎 Testing a sample query...")
    query = "What is the late payment charge?"
    results = query_vectorstore(vectorstore, query)

    print(f"\nTop 3 results for: '{query}'\n")
    print(format_results(results))


if __name__ == "__main__":
    main()
