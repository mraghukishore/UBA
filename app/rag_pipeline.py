from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ─────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"

print("🔌 Connecting to ChromaDB...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("🧠 Loading llama3 model...")
llm = OllamaLLM(model=LLM_MODEL)

# ── Prompt Template ────────────────────────────────────────
prompt = PromptTemplate.from_template("""
You are a helpful customer service assistant for Deccan Power & Gas Utilities Ltd.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I'm sorry, I don't have that information.
Please contact our helpline at 1800-XXX-XXXX."

Context:
{context}

Customer Question: {question}

Answer:""")

# ── Helper to format retrieved docs ───────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Modern RAG Chain ───────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

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