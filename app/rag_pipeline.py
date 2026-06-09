from app.shared import build_rag_chain

print("🔌 Connecting to ChromaDB...")
print("🧠 Loading llama3 model...")
rag_chain = build_rag_chain()

print("\n✅ Billing Assistant is ready!")
print("=" * 50)
print("Type your question below. Type 'exit' to quit.")
print("=" * 50)

# ── Interactive loop ───────────────────────────────────────
while True:
    question = input("\n💬 Your question: ").strip()
    if question.lower() == "exit":
        print("Goodbye! 👋")
        break
    if not question:
        continue

    print("\n⏳ Thinking...")
    answer = rag_chain.invoke(question)
    print(f"\n🤖 Answer:\n{answer}")
