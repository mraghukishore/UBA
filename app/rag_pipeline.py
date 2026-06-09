import sys
import logging
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"


def build_rag_chain(chroma_path, embed_model, llm_model):
    """Initialise the retriever, LLM and RAG chain.

    Raises clear errors when external services (Ollama, ChromaDB) are
    unreachable or misconfigured.
    """
    # ── Connect to ChromaDB ────────────────────────────────
    try:
        embeddings = OllamaEmbeddings(model=embed_model)
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to ChromaDB at '{chroma_path}' "
            f"(embed model='{embed_model}'). Is Ollama running? Error: {exc}"
        ) from exc

    # ── Load the LLM ───────────────────────────────────────
    try:
        llm = OllamaLLM(model=llm_model)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to load LLM model '{llm_model}' via Ollama. "
            f"Is Ollama running and the model pulled? Error: {exc}"
        ) from exc

    # ── Prompt Template ────────────────────────────────────
    prompt = PromptTemplate.from_template("""
You are a helpful customer service assistant for Deccan Power & Gas Utilities Ltd.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I'm sorry, I don't have that information.
Please contact our helpline at 1800-XXX-XXXX."

Context:
{context}

Customer Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    print("🔌 Connecting to ChromaDB and loading llama3...")
    try:
        rag_chain = build_rag_chain(CHROMA_PATH, EMBED_MODEL, LLM_MODEL)
    except ConnectionError as exc:
        logger.error("Startup failed: %s", exc)
        sys.exit(1)

    print("\n✅ Billing Assistant is ready!")
    print("=" * 50)
    print("Type your question below. Type 'exit' to quit.")
    print("=" * 50)

    # ── Interactive loop ───────────────────────────────────
    try:
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
            except EOFError:
                break

            if question.lower() == "exit":
                print("Goodbye! 👋")
                break
            if not question:
                continue

            print("\n⏳ Thinking...")
            try:
                answer = rag_chain.invoke(question)
                print(f"\n🤖 Answer:\n{answer}")
            except Exception as exc:
                logger.error("Failed to generate answer: %s", exc)
                print(
                    "\n⚠️  Something went wrong while generating the answer. "
                    "Please try again or type 'exit' to quit."
                )
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
