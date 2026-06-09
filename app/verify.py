from app.shared import get_vectorstore

print("🔍 Connecting to ChromaDB...")
vectorstore = get_vectorstore()

total = vectorstore._collection.count()
print(f"✅ Total chunks in ChromaDB: {total}")

print("\n🔎 Testing a sample query...")
query = "What is the late payment charge?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 3 results for: '{query}'\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:200])
    print()
