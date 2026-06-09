"""Shared utilities for embeddings, vectorstore, prompts, and RAG chain."""

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import CHROMA_PATH, EMBED_MODEL, LLM_MODEL, RETRIEVER_K

PROMPT_TEMPLATE = """
You are a helpful customer service assistant for Deccan Power & Gas Utilities Ltd.
Answer the customer's question using ONLY the context provided below.
Be concise, friendly and professional.
If the answer is not in the context, say "I'm sorry, I don't have that information.
Please contact our helpline at 1800-XXX-XXXX for assistance."

Context:
{context}

Customer Question: {question}

Answer:"""


def get_embeddings():
    """Return an OllamaEmbeddings instance using the configured model."""
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_vectorstore():
    """Return a Chroma vectorstore connected to the persisted DB."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )


def format_docs(docs):
    """Join a list of documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_prompt():
    """Return the shared PromptTemplate for the RAG chain."""
    return PromptTemplate.from_template(PROMPT_TEMPLATE)


def build_rag_chain(vectorstore=None):
    """Build and return the full RAG chain.

    If *vectorstore* is ``None`` a new one is created via :func:`get_vectorstore`.
    """
    if vectorstore is None:
        vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = OllamaLLM(model=LLM_MODEL)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | get_prompt()
        | llm
        | StrOutputParser()
    )
    return chain
