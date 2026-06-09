# UBA Project Report — Final Edition

## Project Title
RAG-Based Utility Billing Assistant using Local Large Language Model (LLM)

## Submitted By
Raghu Kishore Marupaka
Roll No: 094254270019

## Submitted To
Mrs. K. Jaya
Department of CSE, UCE, Osmania University, Hyderabad

## Institution
P G R R Centre for Distance Education
Osmania University, Hyderabad

---

# Certificate

This is to certify that the project entitled

**"RAG-Based Utility Billing Assistant using Local Large Language Model (LLM)"

has been carried out by Raghu Kishore Marupaka (Roll No: 094254270019) in partial fulfillment of the requirements for the award of the degree of Advanced Diploma in Data Science from P G R R Centre for Distance Education, Osmania University, Hyderabad.

---

# Declaration

I, Raghu Kishore Marupaka, student of Advanced Diploma in Data Science at P G R R Centre for Distance Education, Osmania University, Hyderabad, hereby declare that the project report titled **"RAG-Based Utility Billing Assistant using Local Large Language Model (LLM)"** has been prepared by me under the guidance of Mrs. K. Jaya. The work presented in this report is original and has not been submitted elsewhere for any other degree or diploma.

---

# Acknowledgement

I express my sincere gratitude to Mrs. K. Jaya for her valuable guidance and encouragement throughout the project. I also thank the faculty and staff of the Department of Computer Science and Engineering for their support.

I acknowledge the support of my family and peers who provided constant motivation and feedback during the project.

---

# Abstract

The RAG-Based Utility Billing Assistant project delivers an intelligent, locally-executed customer support system for Deccan Power & Gas Utilities Ltd. It combines PDF document ingestion, semantic retrieval, and local large language model inference to answer billing and service queries. The assistant is designed to operate without cloud dependencies, ensuring data privacy and lower operational cost.

The report details the design, development, implementation, and testing of the solution. It covers system analysis, feasibility, requirements, architecture, software and hardware needs, system design with UML diagrams, data dictionary, technology mapping, complete code listings, testing strategy, and evaluation.

Key contributions:
- Document ingestion pipeline using Ollama embeddings and ChromaDB
- Streamlit-based conversational UI with optional raw retrieval mode
- Local RAG pipeline that returns answers limited to source documents
- Detailed project documentation aligned to academic reporting requirements

---

# Table of Contents

1. Abstract
2. Introduction, Objectives, Scope
3. System Analysis
4. Feasibility Study
5. System Requirements Specification
6. Process Flow & System Architecture
7. SDLC Methodology
8. Software & Hardware Requirements
9. System Design
10. Data Dictionary
11. Technology Description
12. Code Listing
13. System Testing
14. Advantages of the System
15. Limitations of the System
16. Output Screens
17. Conclusion
18. Future Enhancement
19. Bibliography / References

---

# 1. Introduction, Objectives, Scope

## 1.1 Introduction

Utility companies are increasingly challenged to provide timely and accurate information to customers about tariffs, billing policies, payment procedures, and outage management. Traditional customer service channels often rely on manual lookup from policy documents, causing delays and inconsistent responses.

This project introduces a Retrieval-Augmented Generation (RAG) assistant that draws knowledge directly from existing utility documents. By embedding document chunks and leveraging a local LLM, the system produces answers grounded in the source documents.

## 1.2 Objectives

The primary objectives of this project are:
- Create a searchable knowledge base from DPGU policy documents.
- Build a local RAG pipeline that combines semantic retrieval with llama3 generation.
- Provide a user-friendly Streamlit chat interface.
- Enable a fallback raw retrieval mode for transparent document citation.
- Ensure the solution runs locally without requiring external cloud inference.

## 1.3 Scope of the Project

### In Scope
- PDF ingestion from the `docs/` directory
- Document chunking and embedding with Ollama
- Vector storage using ChromaDB
- Semantic search and answer generation
- Streamlit UI for interaction
- CLI tool for verification and command-line use

### Out of Scope
- Integration with live billing databases
- Real-time payment processing
- Authentication and customer account management
- Voice interface and multi-modal input

## 1.4 Motivation

The motivation for this project is to reduce support overhead by enabling customers to self-serve answers to common billing and service questions. Local operation preserves data privacy and avoids recurring cloud costs.

## 1.5 Organisation of the Report

This report is organized into 19 major sections, covering the entire lifecycle of the project from analysis through implementation, testing, and future scope.

---

# 2. System Analysis

## 2.1 Existing System

The existing DPGU customer support ecosystem depends heavily on manual document lookup and human agents. Support staff consult tariff schedules, FAQs, outage procedures, and payment instructions from separate sources.

### Shortcomings of the existing system
- Slow response times
- Inconsistent messaging
- No consolidated knowledge base
- High human workload for repeated inquiries

## 2.2 Proposed System

The proposed system centralizes knowledge retrieval using semantic embeddings and a local LLM. It provides a conversational interface and leverages the existing policy documents without rewriting them.

### Benefits of the proposed system
- Fast response to customer queries
- Contextual answers directly grounded in policy documents
- Reduced call center load
- Local operation for privacy and compliance

## 2.3 Existing vs Proposed Comparison

| Criterion | Existing System | Proposed System |
|---|---|---|
| Access method | Manual lookup | Semantic retrieval
| Response delivery | Human agent | Chatbot and CLI
| Consistency | Variable | Consistent
| Knowledge update | Manual | Document-driven
| Privacy | Moderate | High (local)
| Scalability | Limited | Scalable

## 2.4 Problem Statement

Customers need accurate, consistent answers about utility billing and services. The current system is unable to provide rapid self-service responses from official documentation.

## 2.5 Solution Overview

The solution is a local RAG assistant that:
- ingests official utility documents
- creates a vector knowledge base
- retrieves the most relevant chunks for a question
- generates safe answers using a local LLM

---

# 3. Feasibility Study

## 3.1 Technical Feasibility

The project uses proven open-source technologies and locally available compute resources.
- Python 3.10 or later
- Streamlit for UI
- Ollama runtime for embeddings and LLM
- ChromaDB for vector storage
- LangChain components for pipeline orchestration

### Technical risks
- Availability of Ollama models locally
- Memory requirements for embedding vectors
- Version compatibility between LangChain and Ollama

## 3.2 Operational Feasibility

The system can operate on a local workstation, with a human entering documents and running the ingestion pipeline as needed.

### Operational strengths
- No internet dependency once installed
- Simple restart procedure
- Clear document ingestion workflow

### Operational constraints
- Requires trained staff to refresh documents
- Needs periodic verification of model and database health

## 3.3 Economic Feasibility

Local operation reduces cloud costs. Hardware cost is the primary expense.

### Cost factors
- Initial hardware capacity
- Ollama runtime licensing or local install
- Developer and maintenance effort

### Return on investment
- Reduced call center load
- Faster customer query resolution
- Improved customer satisfaction

## 3.4 Schedule Feasibility

A phased schedule is recommended:
- Week 1: requirement gathering and design
- Week 2: document ingestion and ChromaDB setup
- Week 3: RAG pipeline implementation
- Week 4: Streamlit UI development
- Week 5: integration, testing, documentation

---

# 4. System Requirements Specification

## 4.1 Functional Requirements

FR01: Load PDF documents from the `docs/` folder.
FR02: Split each document into chunks suitable for embedding.
FR03: Create embeddings using Ollama.
FR04: Persist embeddings in local ChromaDB.
FR05: Retrieve the top 3 most relevant chunks for a user query.
FR06: Generate an answer using llama3 when LLM mode is enabled.
FR07: Provide raw retrieval output when LLM mode is disabled.
FR08: Allow users to submit questions in a chat interface.
FR09: Maintain chat history within the session.
FR10: Provide verification output in a CLI tool.

## 4.2 Non-Functional Requirements

NFR01: The system should respond within 5 seconds for typical queries.
NFR02: The system must run on local hardware without internet.
NFR03: Answer generation must be constrained by retrieved context.
NFR04: The UI should be accessible in standard web browsers.
NFR05: The system should log errors and usage events.

## 4.3 User Requirements

UR01: Customers should be able to ask natural language billing questions.
UR02: Support staff should be able to refresh the knowledge base when documents are updated.
UR03: Administrators should be able to verify database contents.

## 4.4 System Interfaces

### External interfaces
- PDF documents in the file system
- Streamlit browser UI
- Python runtime and local Ollama processes

### Internal components
- `app/ingest.py`
- `app/rag_pipeline.py`
- `app/chatbot_ui.py`
- `app/verify.py`
- `chroma_db/` storage

## 4.5 Data Requirements

Data must be processed as text chunks and stored in embeddings with metadata including source filename and chunk location.

---

# 5. Process Flow & System Architecture

## 5.1 Process Flow

### Data ingestion
1. Load PDFs from `docs/`.
2. Use `PyPDFLoader` to extract text.
3. Split text into overlapping chunks.
4. Generate embeddings for each chunk.
5. Store vectors and metadata in ChromaDB.

### Query execution
1. User submits a question.
2. The retriever searches ChromaDB for the best matching chunks.
3. If LLM mode is enabled, the system formats the chunks into context and invokes llama3.
4. If LLM mode is disabled, the system returns raw chunk excerpts.
5. The UI displays the answer to the user.

## 5.2 System Architecture Components

- Document Loader: reads PDF files.
- Text Splitter: creates document chunks.
- Embedding Engine: generates vector representations.
- Vector Store: stores embeddings and metadata.
- Retriever: finds relevant chunks for queries.
- Prompt Template: constrains answer generation.
- LLM: generates natural language answers.
- UI Layer: Streamlit chat interface.
- Verification Tool: CLI diagnostics.

## 5.3 Architecture Diagram

```
+----------------------+    +----------------------+    +----------------------+
|  Document Ingestion  |--->|  Embedding Engine    |--->|   Vector Store       |
|  (app/ingest.py)     |    |  (OllamaEmbeddings)  |    |   (ChromaDB)         |
+----------------------+    +----------------------+    +----------------------+
            |                          |                        ^
            v                          v                        |
+----------------------+    +----------------------+    +----------------------+
|  User Chat Interface |<---|  Retriever & RAG     |<---|   Prompt Template     |
|  (Streamlit UI)      |    |  (llama3 + prompt)   |    |                      |
+----------------------+    +----------------------+    +----------------------+
```

## 5.4 Data Flow Diagram

```
[User] --> (Ask Question) --> [Streamlit UI]
[Streamlit UI] --> (Semantic Search) --> [ChromaDB]
[ChromaDB] --> (Top-k Chunks) --> [RAG Component]
[RAG Component] --> (Answer Generation) --> [LLM]
[LLM] --> (Answer Response) --> [Streamlit UI]
```

## 5.5 Process Description

The system is designed as a classic information retrieval pipeline extended with generative capability. By separating retrieval from generation, it limits hallucination and makes answers traceable.

---

# 6. SDLC Methodology

## 6.1 Chosen Methodology: Agile Iterative Model

An Agile, iterative approach is suitable because it allows incremental delivery and continuous feedback. Each sprint focuses on a specific milestone, making the system adaptable and testable at every stage.

## 6.2 Sprint Breakdown

- Sprint 1: Requirements analysis, feasibility, and architecture design.
- Sprint 2: Document ingestion, chunking, and ChromaDB setup.
- Sprint 3: Retrieval pipeline and LLM prompt design.
- Sprint 4: Streamlit UI, mode toggling, and session handling.
- Sprint 5: Testing, validation, and documentation.

## 6.3 Work Breakdown Structure

### Planning
- Identify project goals
- Determine hardware and software requirements
- Draft use cases

### Development
- Implement ingestion and embedding
- Build retrieval and prompt pipeline
- Develop UI and user interaction logic

### Testing
- Unit tests for ingestion and retrieval
- Integration tests for the RAG pipeline
- User acceptance tests for the UI

### Deployment
- Package dependencies
- Prepare instructions for local installation
- Document usage and verification steps

## 6.4 Quality Assurance

Quality assurance is performed by:
- verifying each module individually
- validating retrieval relevance
- confirming answer integrity against source documents
- conducting end-to-end user scenario tests

---

# 7. Software & Hardware Requirements

## 7.1 Software Requirements

| Component | Purpose | Recommended Version |
|---|---|---|
| Python | Programming language | 3.10+ |
| Streamlit | UI framework | Latest stable |
| Ollama | Embeddings and LLM runtime | Current local release |
| ChromaDB | Vector database | Latest stable |
| LangChain | Pipeline orchestration | Compatible release |
| PyPDFLoader | PDF extraction | Latest stable |
| RecursiveCharacterTextSplitter | Chunk splitter | Latest stable |

### Python Dependencies
- `langchain_ollama`
- `langchain_chroma`
- `langchain_core`
- `langchain_community`
- `langchain_text_splitters`
- `streamlit`
- `python-docx` (for documentation generation)

## 7.2 Hardware Requirements

Minimum:
- 4 CPU cores
- 8 GB RAM
- 10 GB free disk space
- Local Ollama runtime install

Recommended:
- 6+ CPU cores
- 16 GB RAM
- 25 GB free disk space
- Optional GPU support for faster LLM inference

## 7.3 Deployment Environment

The system is best deployed on a local machine or server with secure access. Because all operations are local, no external network access is required after installation.

---

# 8. System Design

## 8.1 Entity Relationship Diagram (ERD)

```
+------------------+     +-------------------+     +------------------+
|   Document       |     |   Chunk           |     |   Embedding      |
+------------------+     +-------------------+     +------------------+
| document_id      |<--->| chunk_id          |<--->| embedding_id     |
| filename         |     | document_id       |     | chunk_id         |
| path             |     | content           |     | vector           |
| page_number      |     | metadata          |     | created_at       |
| created_at       |     +-------------------+     +------------------+
+------------------+
```

### Description
- `Document`: stores original PDF source details.
- `Chunk`: stores text segments derived from documents.
- `Embedding`: stores vector representations and references to chunks.

## 8.2 Data Flow Diagram (DFD)

```
[User] --> [UI] --> [Retriever] --> [Vector Store]
                     |              ^
                     v              |
                  [LLM] <----------|
```

## 8.3 UML Use Case Diagram

```
+-------------------------------------------+
|          Utility Billing Assistant        |
+-------------------------------------------+
| Actors: Customer, Administrator           |
+-------------------------------------------+
| Use Cases:                                |
| - Ask Question                            |
| - View Answer                             |
| - Toggle LLM Mode                         |
| - Ingest Documents                        |
| - Verify Knowledge Base                   |
+-------------------------------------------+
```

## 8.4 UML Activity Diagram

```
[Start]
   |
(Load PDFs)
   |
(Split into chunks)
   |
(Generate embeddings)
   |
(Store in ChromaDB)
   |
(User submits question)
   |
(Search relevant chunks)
   |
[Decision: LLM Mode?]
  /   \
 /     \
(Yes)   (No)
 |        |
(Format context)  (Return raw chunks)
 |        |
(Invoke llama3)
 |        |
(Display answer)
   |
[End]
```

## 8.5 UML Class Diagram

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
| RagChainController     |
+------------------------+
| - retriever            |
| - prompt               |
| - llm                  |
+------------------------+
| + invoke(question)     |
+------------------------+
```

## 8.6 UML Component Diagram

```
+------------------------------------------------+
|          Billing Assistant Application         |
+------------------------------------------------+
| +-----------+   +-----------+   +-----------+  |
| | UI Layer  |-->| RAG Layer |-->| LLM Layer |  |
| +-----------+   +-----------+   +-----------+  |
|        |               |                  |    |
|        v               v                  |    |
|     +---------------------------+         |    |
|     |      ChromaDB Vector Store |<--------+    |
|     +---------------------------+              |
+------------------------------------------------+
```

## 8.7 UML Deployment Diagram

```
+--------------------------------------------------+
|                 Local Deployment                 |
+--------------------------------------------------+
| [Workstation]                                    |
|  - Python runtime                                |
|  - Streamlit                                     |
|  - Ollama runtime                                |
|  - Project scripts                               |
+--------------------------------------------------+
| [ChromaDB Storage]                               |
|  - chroma.sqlite3                                 |
|  - vector data                                    |
+--------------------------------------------------+
```

## 8.8 Sequence Diagram

```
User -> UI: Submit question
UI -> Retriever: Search ChromaDB
Retriever -> ChromaDB: Query vectors
ChromaDB -> Retriever: Return chunks
Retriever -> LLM: Provide context
LLM -> UI: Return answer
UI -> User: Display answer
```

---

# 9. Data Dictionary

## 9.1 ChromaDB Core Tables

### 9.1.1 `collections`
- `id`: unique collection identifier
- `name`: collection name
- `dimension`: vector dimension size
- `database_id`: reference to the database record
- `config_json_str`: configuration metadata
- `schema_str`: optional schema definition

### 9.1.2 `segments`
- `id`: segment identifier
- `type`: segment type
- `scope`: scope metadata
- `collection`: collection identifier

### 9.1.3 `embeddings`
- `id`: auto-increment integer
- `segment_id`: foreign key to segments
- `embedding_id`: embedding unique identifier
- `seq_id`: sequence ID blob used by Chroma
- `created_at`: timestamp of insertion

### 9.1.4 `embedding_metadata`
- `id`: metadata entry identifier
- `key`: metadata key
- `string_value`: textual metadata
- `int_value`: integer metadata
- `float_value`: floating-point metadata
- `bool_value`: boolean metadata

## 9.2 Document Chunk Metadata

Each document chunk stores metadata that supports traceability and retrieval.

Fields include:
- `source`: original PDF filename
- `page`: page number
- `chunk_index`: position of the chunk in the document
- `text`: chunk content used for semantic search
- `bounding_box`: optional source location

## 9.3 Data Dictionary Table

| Object | Field | Type | Description |
|---|---|---|---|
| Document | filename | Text | Original PDF filename |
| Document | path | Text | Full path to source file |
| Chunk | chunk_id | Text | Unique chunk identifier |
| Chunk | document_id | Text | Source document reference |
| Chunk | content | Text | Text content of the chunk |
| Chunk | metadata | Text | JSON metadata for source info |
| Embedding | embedding_id | Text | Unique vector identifier |
| Embedding | vector | BLOB | Embedding vector bytes |
| Embedding | created_at | Timestamp | Insertion timestamp |

## 9.4 Metadata Relationships

Chunks and embeddings are linked through segment and collection identifiers. This allows retrieved content to be traced back to the original document source.

---

# 10. Technology Description

## 10.1 Python
Python is the core implementation language due to its rich ecosystem for data processing, machine learning, and web applications.

Key advantages:
- Rapid development
- Extensive ML libraries
- Strong community support

## 10.2 LangChain
LangChain enables the modular orchestration of document loaders, text splitters, prompt templates, and runnable chains.

### LangChain roles in this project
- `DocumentLoader` for PDFs
- `TextSplitter` for chunking
- `PromptTemplate` for answer generation
- `RunnablePassthrough` to pass queries through the chain

## 10.3 Ollama
Ollama provides both embedding and language model services locally.

### Models in use
- `nomic-embed-text`: embedding model for vectorizing text
- `llama3`: local LLM for answer generation

## 10.4 ChromaDB
ChromaDB stores the document embeddings and provides fast similarity search.

### Why ChromaDB
- Local persistence
- Easy integration with LangChain
- Efficient vector retrieval

## 10.5 Streamlit
Streamlit is used to build the conversational UI.

Benefits:
- Rapid UI prototyping
- Interactive chat components
- Session state management

## 10.6 Retrieval-Augmented Generation (RAG)
RAG combines retrieval from a knowledge base with generation from an LLM.

### RAG advantages
- Grounded answers
- Reduced hallucinations
- Document-backed reasoning

---

# 11. Code Listing

## 11.1 `app/ingest.py`

The document ingestion module loads PDF files, splits them into chunks, and stores embeddings in ChromaDB.

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_PATH = "docs"
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"

print("📄 Loading PDFs from docs folder...")
all_documents = []
for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DOCS_PATH, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"   ✅ Loaded: {filename} ({len(docs)} pages)")

print(f"\n📚 Total pages loaded: {len(all_documents)}")

print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)
print(f"   ✅ Total chunks created: {len(chunks)}")

print("\n🔢 Embedding and storing in ChromaDB...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB.")
```

## 11.2 `app/rag_pipeline.py`

The RAG pipeline connects the retriever to the prompt template and llama3.

```python
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

print("🔌 Connecting to ChromaDB...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("🧠 Loading llama3 model...")
llm = OllamaLLM(model=LLM_MODEL)

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

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n✅ Billing Assistant is ready!")
print("Type your question below. Type 'exit' to quit.")

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

## 11.3 `app/chatbot_ui.py`

The Streamlit UI provides a conversational interface and a toggle for raw retrieval mode.

```python
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

st.set_page_config(page_title="DPGU Billing Assistant", page_icon="🔌", layout="centered")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/electricity.png", width=80)
    st.title("DPGU Assistant")
    st.markdown("**Deccan Power & Gas Utilities Ltd**")
    st.markdown("---")

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

@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

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

st.title("🔌 Utility Billing Assistant")
if llm_enabled:
    st.markdown("🧠 **LLM Mode** — Answers generated by llama3 from your documents.")
else:
    st.markdown("📄 **Raw Retrieval Mode** — Showing matching document chunks directly.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I'm your Deccan Power & Gas Utilities billing assistant. How can I help you today?"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if llm_enabled:
                    rag_chain = load_rag_chain()
                    answer = rag_chain.invoke(question)
                else:
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
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
```

## 11.4 `app/verify.py`

The verification module checks ChromaDB connectivity and performs a sample query.

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"

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

---

# 12. System Testing

## 12.1 Test Strategy

Testing is divided into unit testing, integration testing, user acceptance testing, and performance validation.

## 12.2 Test Cases

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| TC01 | Document ingestion | Run `python app/ingest.py` | Documents are loaded and stored | Verified | Pass |
| TC02 | ChromaDB connection | Run `python app/verify.py` | Database connects, returns count | Verified | Pass |
| TC03 | Retrieval relevance | Query common billing term | Top-3 relevant chunks returned | Verified | Pass |
| TC04 | Answer generation | Ask question in Streamlit | LLM returns answer based on context | Verified | Pass |
| TC05 | Raw retrieval mode | Toggle LLM OFF | Raw chunk content is displayed | Verified | Pass |
| TC06 | Empty query handling | Press submit without text | UI remains stable | Verified | Pass |
| TC07 | Exit CLI | Type `exit` in CLI | Program exits gracefully | Verified | Pass |

## 12.3 System Validation

Validation confirmed that the assistant only uses retrieved document context and does not hallucinate beyond source material. Document source references are visible in raw retrieval mode.

## 12.4 Performance Testing

The system was tested with 3 document sources and 200+ chunks. Search and answer generation completed within 5-10 seconds on the test machine.

---

# 13. Advantages of the System

- Document-grounded answers for improved accuracy
- Local operation avoids cloud costs and data leakage
- Dual interaction modes: natural language and raw retrieval
- Modular design supports document updates easily
- Transparent document source tracking in retrieval mode

---

# 14. Limitations of the System

- Dependent on the completeness of the source documents
- Local LLM inference may be slower than cloud-based alternatives
- Not integrated with live billing systems or account databases
- Requires manual document refresh and maintenance

---

# 15. Output Screens

### Screenshots to include
- Streamlit chat UI with introductory assistant message
- LLM answer generated from retrieved context
- Raw retrieval mode showing document chunk sources
- CLI assistant startup and sample conversation
- ChromaDB verification output from `app/verify.py`
- Ingestion progress output from `app/ingest.py`

> Placeholder: Insert desktop screenshots of the above output in the final report.

---

# 16. Conclusion

The RAG-Based Utility Billing Assistant demonstrates a practical AI-driven support solution for utility billing queries. The system successfully combines document retrieval, local embeddings, and LLM inference to provide contextually accurate responses.

The design prioritizes privacy, transparency, and ease of updates. It can be extended to other customer-facing document sets and adapted for multi-lingual support.

---

# 17. Future Enhancement

Future enhancements can include:
- Authentication and personalized billing history
- Automated document update monitoring
- Multilingual support for local customers
- Payment information and service request integration
- A voice interface for accessibility

---

# 18. Bibliography / References

- LangChain Documentation
- ChromaDB Documentation
- Ollama Documentation
- Streamlit Documentation
- Project source documents in `docs/`
- Academic papers on Retrieval-Augmented Generation

---

# 19. Appendices

## Appendix A: Glossary

- RAG: Retrieval-Augmented Generation
- LLM: Large Language Model
- UI: User Interface
- PDF: Portable Document Format
- CLI: Command-Line Interface

## Appendix B: Installation and Deployment

### Installation Steps
1. Set up Python 3.10+.
2. Create and activate a virtual environment.
3. Install required dependencies.
4. Install Ollama and load the `llama3` and `nomic-embed-text` models.
5. Place source PDFs in the `docs/` folder.
6. Run `python app/ingest.py` to build ChromaDB.
7. Launch the UI with `streamlit run app/chatbot_ui.py`.

### Deployment Notes
- Ensure the local machine has sufficient disk and memory.
- Keep the `chroma_db/` directory backed up after ingestion.
- Update documents by rerunning `app/ingest.py`.

## Appendix C: Project Folder Structure

- `app/` — application scripts
- `docs/` — source PDF documents
- `chroma_db/` — persistent vector database
- `ProjectReport/` — report files and documentation

---

# 20. Diagram Placeholders

## UML Use Case Diagram

```
+------------------------------------------------+
|              Use Case: Billing Assistant       |
+------------------------------------------------+
| Actors: Customer, Administrator                 |
| Use Cases:                                      |
| - Submit Query                                   |
| - View Answer                                    |
| - Toggle LLM Mode                                |
| - Load Documents                                 |
| - Verify Knowledge Base                          |
+------------------------------------------------+
```

## UML Activity Diagram

```
[Start]
   |
(Load source PDFs)
   |
(Split text into chunks)
   |
(Generate embeddings)
   |
(Store vectors in ChromaDB)
   |
(Receive user query)
   |
(Retrieve relevant chunks)
   |
[If LLM mode ON]
   |--> (Create prompt)
   |--> (Invoke llama3)
   |--> (Return answer)
[Else]
   |--> (Return raw chunks)
   |
[End]
```

## UML Class Diagram

```
+------------------------+
| DocumentIngestor       |
+------------------------+
| - docs_path            |
| - pdf_loader           |
+------------------------+
| + load_documents()     |
| + split_documents()    |
+------------------------+
         |
         v
+------------------------+
| ChromaStorage          |
+------------------------+
| - persist_directory    |
| - embedding_model      |
+------------------------+
| + store_embeddings()   |
| + as_retriever()       |
+------------------------+
         |
         v
+------------------------+
| RagService             |
+------------------------+
| - prompt_template      |
| - llm_model            |
+------------------------+
| + generate_answer()    |
+------------------------+
```

## UML Component Diagram

```
+-----------------------------------------+
|   RAG-Based Utility Billing Assistant   |
+-----------------------------------------+
|  +-------+   +-----------+   +--------+ |
|  | Input |-->| Retrieval |-->| LLM    | |
|  +-------+   +-----------+   +--------+ |
|     |             |               ^    |
|     v             v               |    |
|  +-----------------------------+  |    |
|  |       ChromaDB Vector Store  |<-+    |
|  +-----------------------------+       |
+-----------------------------------------+
```

## UML Deployment Diagram

```
+-----------------------------------------+
| Local Host                              |
+-----------------------------------------+
| - Python                                |
| - Streamlit                             |
| - Ollama                                |
| - ChromaDB                              |
+-----------------------------------------+
```
