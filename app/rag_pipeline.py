from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ─────────────────────────────────────────────────
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"

PROMPT_TEMPLATE = """
You are a helpful customer service assistant for Deccan Power & Gas Utilities Ltd.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I'm sorry, I don't have that information.
Please contact our helpline at 1800-XXX-XXXX."

Context:
{context}

Customer Question: {question}

Answer:"""


# ── Helper to format retrieved docs ───────────────────────
def format_docs(docs):
    """Join document page_content fields with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(chroma_path=CHROMA_PATH, embed_model=EMBED_MODEL, llm_model=LLM_MODEL):
    """Build and return a RAG chain connected to ChromaDB and an Ollama LLM."""
    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model=llm_model)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # ── Modern RAG Chain ───────────────────────────────────────
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def run_interactive_loop(rag_chain):
    """Run a terminal-based Q&A loop using the given RAG chain."""
    print("\n✅ Billing Assistant is ready!")
    print("=" * 50)
    print("Type your question below. Type 'exit' to quit.")
    print("=" * 50)

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


def main():
    print("🔌 Connecting to ChromaDB...")
    print("🧠 Loading llama3 model...")
    rag_chain = build_rag_chain()
    run_interactive_loop(rag_chain)


if __name__ == "__main__":
    main()
