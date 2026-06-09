# UBA Project Report — Expanded Version

## 1. Abstract
This project implements a local Retrieval-Augmented Generation (RAG) billing assistant for Deccan Power & Gas Utilities Ltd (DPGU). The solution ingests customer-facing utility documents into ChromaDB, embeds them using Ollama embeddings, and serves answers through an interactive Streamlit chatbot UI or command-line assistant backed by llama3. The system demonstrates how modern vector search and local LLM inference can deliver fast, contextual customer support for tariff, billing, outage, and payment queries.

## 2. Introduction, Objectives, Scope
### 2.1 Introduction
The DPGU Billing Assistant is designed to help customers quickly find answers to electricity and gas billing questions using existing policy documents. It combines document ingestion, semantic search, and a local language model to produce concise, accurate answers from the company knowledge base.

### 2.2 Objectives
- Build a searchable knowledge base from PDF policy and tariff documents.
- Implement a local RAG pipeline using Ollama embeddings and llama3.
- Provide both a Streamlit chat UI and a CLI assistant for user interaction.
- Keep all processing local to preserve data privacy.

### 2.3 Scope
In scope:
- Loading official DPGU PDF documents into ChromaDB.
- Vectorizing document chunks for semantic search.
- Generating context-aware answers through llama3.
- Displaying chat-based results via Streamlit.

Out of scope:
- Real-time billing computation.
- Customer login or authentication.
- Payment gateway integration.

## 3. System Analysis
### 3.1 Existing System
The existing paper-based and static FAQ setup requires manual lookup of tariff schedules, payment policies, and outage procedures. Customer service staff frequently handle repeated questions without an automated assistant, causing delays and inconsistent responses.

### 3.2 Proposed System
The proposed RAG assistant automates retrieval from existing documents and provides consistent answers to customer queries. It supports:
- semantic search over policy PDFs
- an intelligent answer generation pipeline
- local model inference with no cloud dependency

### 3.3 Existing vs Proposed Comparison
| Feature | Existing System | Proposed System |
|---|---|---|
| Document access | Manual lookup | Semantic retrieval
| Response time | Slow | Fast
| Consistency | Variable | Consistent
| Availability | Office hours | 24/7 self-service
| Scalability | Limited | High
| Data privacy | Moderate | High (local only)

## 4. Feasibility Study
### 4.1 Technical Feasibility
The system uses open-source tooling and local inference. Required components are:
- Python 3.x
- Streamlit
- Ollama embeddings and LLM runtime
- ChromaDB for vector storage
- PDF loaders and text splitters

### 4.2 Operational Feasibility
The application runs on a local workstation or server. It requires periodic document updates and can be operated by a technical user. The Streamlit UI is intuitive for business users.

### 4.3 Economic Feasibility
Cost drivers include hardware and Ollama runtime usage. Because the system is local, ongoing cloud costs are eliminated. The expected return includes saved support time and reduced call center volume.

## 5. System Requirements Specification
### 5.1 Functional Requirements
- FR1: Ingest PDF documents from `docs/` into ChromaDB.
- FR2: Split documents into chunks for embedding.
- FR3: Store embeddings and document metadata in local ChromaDB.
- FR4: Accept user questions through Streamlit chat UI.
- FR5: Retrieve top-k relevant chunks and generate an answer.
- FR6: Provide a CLI fallback for terminal users.

### 5.2 Non-Functional Requirements
- NFR1: Response time should be under 5 seconds for typical queries.
- NFR2: The system must run without internet access once dependencies are installed.
- NFR3: Answers must be based only on retrieved documents.
- NFR4: The architecture should support adding new documents easily.

### 5.3 Expanded Requirements
- Document source tracing: each returned chunk includes its PDF source.
- Toggle mode: allow switching between raw retrieval mode and full LLM answer generation.
- Error handling for missing documents, corrupted database, and model load failure.

## 6. Process Flow & System Architecture
### 6.1 High-Level Flow
1. Load PDFs from `docs/`.
2. Split text into vector-friendly chunks.
3. Embed chunks with Ollama embeddings.
4. Persist vectors in ChromaDB.
5. Accept user query.
6. Perform semantic search over ChromaDB.
7. Format retrieved chunks into context.
8. Generate answer via llama3 or display raw chunks.

### 6.2 Architecture Components
- Document Ingestion: `app/ingest.py`
- Vector Store: ChromaDB in `chroma_db/`
- Retrieval & RAG Logic: `app/rag_pipeline.py` and `app/chatbot_ui.py`
- Verification: `app/verify.py`
- UI: Streamlit-based chat assistant

## 7. SDLC Methodology
### 7.1 Agile / Iterative Model
The project is best developed using an Agile approach, with short iterations that allow frequent review and incremental delivery.
- Sprint 1: Document ingestion + ChromaDB storage
- Sprint 2: Retrieval pipeline + CLI assistant
- Sprint 3: Streamlit UI + answer generation
- Sprint 4: Testing, reporting, and refinements

## 8. Software & Hardware Requirements
### 8.1 Software Requirements
- Python 3.10+ or later
- Streamlit
- `langchain_ollama`, `langchain_chroma`, `langchain_core`
- `langchain_community.document_loaders`
- `langchain_text_splitters`
- Ollama local runtime with `nomic-embed-text` and `llama3`

### 8.2 Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 8 GB minimum, 16 GB preferred
- Disk: 2 GB free for database and documents
- Optional GPU support for faster Ollama inference

## 9. System Design
### 9.1 ER Diagram
```
+------------------+     +-------------------+     +------------------+
|   Document       |     |   Chunk           |     |   Embedding      |
+------------------+     +-------------------+     +------------------+
| document_id      |<--->| chunk_id          |<--->| embedding_id     |
| filename         |     | document_id       |     | chunk_id         |
| path             |     | text              |     | vector           |
| created_at       |     | metadata          |     | created_at       |
+------------------+     +-------------------+     +------------------+
```

### 9.2 DFD Level 1
```
[User] --> (Ask Question) --> [Chat UI]
[Chat UI] --> (Query Vector Store) --> [ChromaDB]
[ChromaDB] --> (Return Chunks) --> [RAG Service]
[RAG Service] --> (Generate Answer) --> [LLM]
[LLM] --> (Answer) --> [Chat UI]
```

### 9.3 UML Use Case Diagram
```
+-------------------------------------------+
|          Utility Billing Assistant        |
+-------------------------------------------+
| [Customer]                               |
|   - Ask billing questions                 |
|   - View raw retrieval results            |
|   - Toggle LLM mode                       |
+-------------------------------------------+
| [Admin/Developer]                         |
|   - Ingest documents                      |
|   - Verify database contents              |
|   - Update knowledge base                 |
+-------------------------------------------+
```

### 9.4 UML Activity Diagram
```
[Start]
   |
   v
(Load PDFs)
   |
   v
(Split to chunks)
   |
   v
(Embed chunks)
   |
   v
(Store in ChromaDB)
   |
   v
(User submits query)
   |
   v
(Search relevant chunks)
   |
   v
[if LLM mode]
   |--> (Format context)
   |--> (Invoke llama3)
   |--> (Return answer)
[else]
   |--> (Return raw chunks)
   |
   v
[End]
```

### 9.5 UML Class Diagram
```
+------------------------+
| DocumentLoader         |
+------------------------+
| - docs_path            |
| - loader               |
+------------------------+
| + load_documents()     |
+------------------------+
         |
         v
+------------------------+
| ChunkSplitter          |
+------------------------+
| - chunk_size           |
| - overlap              |
+------------------------+
| + split_documents()    |
+------------------------+
         |
         v
+------------------------+
| ChromaManager          |
+------------------------+
| - persist_directory    |
| - embeddings           |
+------------------------+
| + from_documents()     |
| + as_retriever()       |
+------------------------+
         |
         v
+------------------------+
| RAGChain               |
+------------------------+
| - retriever            |
| - prompt               |
| - llm                  |
+------------------------+
| + invoke(question)     |
+------------------------+
```

### 9.6 UML Component Diagram
```
+------------------------------------------------+
|                Billing Assistant App           |
+------------------------------------------------+
|  +-----+   +----------+   +----------+         |
|  | UI  |-->| RAG      |-->| LLM      |         |
|  +-----+   +----------+   +----------+         |
|                 |                        |
|                 v                        |
|             +-------+                    |
|             | Chroma |<------------------|
|             +-------+                    |
+------------------------------------------------+
```

### 9.7 UML Deployment Diagram
```
+----------------------+        +----------------------+
|  Local Workstation   |        |  ChromaDB Storage    |
+----------------------+        +----------------------+
| - Streamlit UI       |        | - chroma.sqlite3     |
| - Python runtime     |        | - collection data    |
| - Ollama runtime     |        +----------------------+
+----------------------+        
```

## 10. Data Dictionary
### 10.1 ChromaDB Schema Overview
The local ChromaDB schema stores document embeddings and metadata for semantic retrieval.

#### 10.1.1 `collections`
- `id`: collection identifier
- `name`: collection name
- `dimension`: embedding vector size
- `database_id`: parent database reference
- `config_json_str`: collection configuration
- `schema_str`: serialized schema details

#### 10.1.2 `embeddings`
- `id`: numeric record ID
- `segment_id`: references the segment containing the vector
- `embedding_id`: unique embedding identifier
- `seq_id`: sequence ID blob
- `created_at`: timestamp of creation

#### 10.1.3 `embedding_metadata`
- `id`: metadata entry ID
- `key`: metadata field name
- `string_value`, `int_value`, `float_value`, `bool_value`: metadata values

#### 10.1.4 `segments`
- `id`: segment identifier
- `type`: segment type label
- `scope`: segment scope
- `collection`: associated collection

#### 10.1.5 `databases`
- `id`: database identifier
- `name`: database name
- `tenant_id`: tenant association

### 10.2 Chunk Metadata
The system ingests PDF pages and splits them into chunks. Each chunk retains source metadata such as:
- `source` file name
- `page_number`
- `chunk_text`
- `split_size`

These metadata values enable traceability from answer back to the original policy document.

## 11. Technology Description
### 11.1 Python
Python is the implementation language for all scripts. It offers robust libraries for ML, vector search, and web UI development.

### 11.2 LangChain
LangChain components coordinate the RAG pipeline:
- document loaders
- text splitters
- prompt templates
- runnable chains

### 11.3 Ollama
Ollama provides both embeddings and the local `llama3` model. The project uses:
- `OllamaEmbeddings(model='nomic-embed-text')`
- `OllamaLLM(model='llama3')`

### 11.4 ChromaDB
ChromaDB stores embeddings and enables fast semantic retrieval. It persists data locally in `chroma_db/chroma.sqlite3`.

### 11.5 Streamlit
Streamlit provides the interactive web-based assistant UI in `app/chatbot_ui.py`.

### 11.6 RAG (Retrieval-Augmented Generation)
RAG combines retrieval from a vector database with LLM generation. It prevents hallucination by constraining responses to retrieved document context.

## 12. Code Listing
### 12.1 `app/ingest.py`
```python
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
```

### 12.2 `app/rag_pipeline.py`
```python
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
```

### 12.3 `app/chatbot_ui.py`
```python
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ─────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"

# ── Page setup ─────────────────────────────────────────────
st.set_page_config(
    page_title="DPGU Billing Assistant",
    page_icon="🔌",
    layout="centered"
)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/electricity.png", width=80)
    st.title("DPGU Assistant")
    st.markdown("**Deccan Power & Gas Utilities Ltd**")
    st.markdown("---")

    # 🔥 LLM Toggle
    st.markdown("### ⚙️ Mode Settings")
    llm_enabled = st.toggle("🧠 Enable LLM (llama3)", value=True)

    if llm_enabled:
        st.success("🟢 LLM ON — llama3 generating answers")
    else:
        st.warning("🟡 LLM OFF — showing raw document chunks only")

    st.markdown("---")
    st.markdown("💡 Ask me anything about:")
    st.markdown("- Electricity & gas tariffs")
    st.markdown("- Bill calculations")
    st.markdown("- Payment options")
    st.markdown("- Outage reporting")
    st.markdown("- Meter issues")
    st.markdown("---")
    st.markdown("🤖 Powered by **llama3** via Ollama")
    st.markdown("🗄️ Knowledge base: **ChromaDB**")
    st.markdown("⚡ Running **100% locally**")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Load embeddings and vectorstore (always needed) ────────
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

# ── Load RAG chain (only when LLM is ON) ───────────────────
@st.cache_resource
def load_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=LLM_MODEL)

    prompt = PromptTemplate.from_template("""
You are a helpful customer service assistant for Deccan Power & Gas Utilities Ltd.
Answer the customer's question using ONLY the context provided below.
Be concise, friendly and professional.
If the answer is not in the context, say "I'm sorry, I don't have that information.
Please contact our helpline at 1800-XXX-XXXX for assistance."

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

# ── Main UI ────────────────────────────────────────────────
st.title("🔌 Utility Billing Assistant")
if llm_enabled:
    st.markdown("🧠 **LLM Mode** — Answers generated by llama3 from your documents.")
else:
    st.markdown("📄 **Raw Retrieval Mode** — Showing matching document chunks directly.")
st.markdown("---")

# ── Chat history ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I'm your Deccan Power & Gas Utilities billing assistant. How can I help you today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────
if question := st.chat_input("Type your question here..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if llm_enabled:
                    # ── LLM ON: full RAG answer ────────────────
                    rag_chain = load_rag_chain()
                    answer = rag_chain.invoke(question)

                else:
                    # ── LLM OFF: raw chunks only ───────────────
                    vectorstore = load_vectorstore()
                    docs = vectorstore.similarity_search(question, k=3)
                    answer = "📄 **Raw document chunks retrieved:**\n\n"
                    for i, doc in enumerate(docs):
                        src = doc.metadata.get('source', 'Unknown')
                        import os
                        src = os.path.basename(src)
                        answer += f"**Chunk {i+1}** — `{src}`\n"
                        answer += f"{doc.page_content}\n\n---\n\n"

                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
```

### 12.4 `app/verify.py`
```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"

print("🔍 Connecting to ChromaDB...")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

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
```

## 13. System Testing
### 13.1 Test Cases
| Test ID | Test Case | Input | Expected Result | Status |
|---|---|---|---|---|
| TC01 | Document Ingestion | Run `python app/ingest.py` | All PDFs load and chunks store in ChromaDB | Pass |
| TC02 | Vector Store Connectivity | Run `python app/verify.py` | ChromaDB connects and returns count | Pass |
| TC03 | Retrieval Search | Query a known billing term | Top-3 relevant chunks returned | Pass |
| TC04 | LLM Answer Generation | Ask billing question in Streamlit | Generated answer from llama3 | Pass |
| TC05 | Raw Retrieval Mode | Toggle LLM OFF in Streamlit | Raw document chunks shown | Pass |
| TC06 | Missing Query | Empty question input | UI remains responsive, no error | Pass |
| TC07 | Exit CLI | Type `exit` in CLI | Program ends gracefully | Pass |

### 13.2 Expanded Testing Notes
- Verified that the assistant does not answer questions outside the available documents by returning the defined fallback message.
- Confirmed that metadata source tracing works in raw retrieval mode.
- Tested Streamlit UI on local browser and validated chat history persistence for the session.

## 14. Advantages of the System
- Fast, contextual retrieval from official documentation.
- No dependency on cloud APIs for inference.
- Dual UI support: CLI and web-based chat.
- Easy update path via document ingestion.
- Built on modular LangChain components.

## 15. Limitations of the System
- Depends on the quality of source documents.
- LLM answers may still require human review for regulatory accuracy.
- Requires local hardware resources for Ollama inference.
- Not yet integrated with live billing databases or payment systems.

## 16. Output Screens
### 16.1 Screenshot Placeholders
- Screenshot 1: Streamlit chat interface with LLM mode enabled
- Screenshot 2: Raw retrieval mode showing retrieved document chunks
- Screenshot 3: CLI assistant startup and sample answer
- Screenshot 4: ChromaDB verification output from `app/verify.py`
- Screenshot 5: Document ingestion progress from `app/ingest.py`

> Insert actual screenshots here before final submission.

## 17. Conclusion
The DPGU Billing Assistant project demonstrates a practical local RAG solution for customer support. It successfully converts document-based knowledge into conversational search results while preserving privacy through local storage and inference.

## 18. Future Enhancement
- Add user authentication and personalized billing summaries
- Extend to multi-lingual support for local customers
- Add automated document update monitoring
- Include real-time tariff calculator integration
- Add offline download of query transcripts for audit

## 19. Bibliography / References
- Project documents located in `docs/`
- Ollama documentation
- ChromaDB documentation
- LangChain documentation
- Streamlit documentation

---

### Appendix A: ChromaDB Table Summary
The database contains core tables for collections, segments, embeddings, and metadata, as well as internal maintenance tables used by ChromaDB.

### Appendix B: Deployment Notes
- Ensure Ollama runtime is installed and accessible.
- Run `python app/ingest.py` after placing PDFs in `docs/`.
- Launch Streamlit with `streamlit run app/chatbot_ui.py`.
- Use `python app/verify.py` to validate the ChromaDB index.
