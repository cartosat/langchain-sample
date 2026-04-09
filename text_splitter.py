# Text Splitters: Breaking documents into smaller chunks
# LLMs have token limits, so large documents must be split into chunks.
# Text splitters control chunk size and overlap to preserve context.

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from dotenv import load_dotenv

load_dotenv()

# Sample long text to split
long_text = """Artificial Intelligence (AI) is intelligence demonstrated by machines.
It includes machine learning, where computers learn from data without explicit programming.
Deep learning is a subset of machine learning using neural networks with many layers.

Natural Language Processing (NLP) enables computers to understand human language.
Applications include chatbots, translation, and sentiment analysis.
Large Language Models like GPT are trained on massive text datasets.

Computer Vision allows machines to interpret visual information from the world.
It powers facial recognition, self-driving cars, and medical image analysis.
Convolutional Neural Networks (CNNs) are commonly used in computer vision tasks."""

# --- RecursiveCharacterTextSplitter (recommended, most common) ---
# Tries to split on paragraphs, then sentences, then words
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # max characters per chunk
    chunk_overlap=50,    # overlap between chunks to preserve context
)
# You do not see visible overlap because RecursiveCharacterTextSplitter 
# prefers clean separator boundaries like newlines/sentences, so your text 
# is split into naturally small chunks without needing repeated overlap text.
chunks = recursive_splitter.split_text(long_text)
print("=== RecursiveCharacterTextSplitter ===")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

# --- CharacterTextSplitter (splits on a single separator) ---
char_splitter = CharacterTextSplitter(
    separator="\n\n",    # split on double newlines (paragraphs)
    chunk_size=300,
    chunk_overlap=0,
)
chunks = char_splitter.split_text(long_text)
print("\n=== CharacterTextSplitter (by paragraph) ===")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

# --- Splitting Document objects (preserves metadata) ---
doc = Document(
    page_content=long_text,
    metadata={"source": "ai-intro.txt", "author": "demo"}
)
split_docs = recursive_splitter.split_documents([doc])
print("\n=== Splitting Documents (metadata preserved) ===")
print(f"Number of document chunks: {len(split_docs)}")
print(f"First chunk metadata: {split_docs[0].metadata}")
