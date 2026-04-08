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
