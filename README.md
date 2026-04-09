# LangChain Concepts - Learning Guide

A collection of minimal Python examples to understand core LangChain concepts. Each file is self-contained and demonstrates one concept.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file and paste your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

---

## Concepts

### 1. Models (`models.py`)

Models are the core of LangChain — they take input and generate output. LangChain primarily supports **Chat Models** (like `ChatOpenAI`) which work with a list of messages (system, human, AI). You configure models with parameters like `temperature` (creativity) and `max_tokens` (response length).

```bash
python models.py
```

**Key takeaway:** Chat models take messages in, return an `AIMessage` out.

---

### 2. Prompts (`prompts.py`)

Prompt templates let you create **reusable, parameterized prompts** with placeholders (e.g., `{topic}`). Instead of hardcoding prompt strings, you define a template once and fill in variables at runtime. `ChatPromptTemplate` is used for chat models and supports system/human message pairs.

```bash
python prompts.py
```

**Key takeaway:** Templates separate prompt structure from data, making prompts reusable.

---

### 3. Output Parsers (`output_parser.py`)

Output parsers **transform raw LLM text into structured data**. `StrOutputParser` extracts the string content from an `AIMessage`. `JsonOutputParser` parses the response into Python dicts/lists, and can use Pydantic models to define the expected schema. Parsers are chained after the model using the `|` operator.

```bash
python output_parser.py
```

**Key takeaway:** Parsers convert unstructured LLM text into structured Python objects (strings, dicts, Pydantic models).

---

### 4. Runnables / LCEL (`runnables.py`)

**LangChain Expression Language (LCEL)** is the way to compose components. Any component that implements `invoke()` is a **Runnable**. You chain them with the `|` (pipe) operator: `prompt | model | parser`. Key runnables include:
- **RunnableLambda** — wraps any Python function as a Runnable
- **RunnablePassthrough** — forwards input unchanged
- **RunnableParallel** — runs multiple chains simultaneously

```bash
python runnables.py
```

**Key takeaway:** LCEL lets you build pipelines by connecting components with `|`, like Unix pipes.

---

### 5. Document Loaders (`document_loader.py`)

Document loaders **read data from various sources** (text files, PDFs, web pages, databases) and return `Document` objects. Each `Document` has two fields: `page_content` (the text) and `metadata` (source info like filename, page number). LangChain has loaders for 100+ data sources.

```bash

```

**Key takeaway:** Loaders standardize data from any source into `Document` objects.

---

### 6. Text Splitters (`text_splitter.py`)

LLMs have token limits, so large documents must be **split into smaller chunks**. `RecursiveCharacterTextSplitter` (recommended) tries to split on natural boundaries (paragraphs → sentences → words). You control `chunk_size` (max characters per chunk) and `chunk_overlap` (shared characters between chunks to preserve context).

Why you may not see visible overlap:
You do not see visible overlap because RecursiveCharacterTextSplitter prefers clean separator boundaries like newlines/sentences, so your text is split into naturally small chunks without needing repeated overlap text.

```bash
python text_splitter.py
```

**Key takeaway:** Splitters break large text into LLM-friendly chunks while preserving context through overlap.

---

### 7. Vector Stores (`vector_store.py`)

Vector stores convert text into **numerical vectors (embeddings)** and store them for similarity search. When you search, your query is also converted to a vector, and the store finds the most similar documents. FAISS is a popular in-memory vector store. This is the foundation of semantic search.

```bash
python vector_store.py
```

**Key takeaway:** Vector stores enable finding documents by meaning (semantic similarity), not just keyword matching.

---

### 8. Retrievers (`retrievers.py`)

Retrievers are a **high-level interface** for fetching relevant documents. A retriever wraps a vector store (or other source) and returns documents for a query. The key pattern is **RAG (Retrieval Augmented Generation)**: retrieve relevant docs → inject them into a prompt → ask the LLM to answer based on the context.

```bash
python retrievers.py
```

**Key takeaway:** Retrievers + LLMs = RAG, where the model answers questions using your own data.

---
