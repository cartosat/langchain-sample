# Vector Stores: Storing and searching document embeddings
# Vector stores convert text into numerical vectors (embeddings) and enable
# similarity search - finding documents most relevant to a query.

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# --- Create sample documents ---
documents = [
    Document(page_content="Python is a popular programming language for data science.", metadata={"topic": "python"}),
    Document(page_content="JavaScript is used for web development and runs in browsers.", metadata={"topic": "javascript"}),
    Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"topic": "ml"}),
    Document(page_content="React is a JavaScript library for building user interfaces.", metadata={"topic": "react"}),
    Document(page_content="Pandas is a Python library for data manipulation and analysis.", metadata={"topic": "python"}),
]

# --- Create embeddings ---
# Embeddings convert text into numerical vectors
embeddings = OpenAIEmbeddings()

# --- Build FAISS vector store from documents ---
vectorstore = FAISS.from_documents(documents, embeddings)
print("=== Vector store created with", len(documents), "documents ===")

# --- Similarity search: find documents most relevant to a query ---
query = "What tools are used for data analysis?"
results = vectorstore.similarity_search(query, k=2)
print(f"\n=== Similarity search for: '{query}' ===")
for i, doc in enumerate(results):
    print(f"\n[{i+1}] (topic: {doc.metadata['topic']})")
    print(f"    {doc.page_content}")

# --- Similarity search with scores (lower = more similar) ---
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
print(f"\n=== Similarity search with scores ===")
for doc, score in results_with_scores:
    print(f"  Score: {score:.4f} | {doc.page_content[:60]}...")

# --- Add more documents to existing store ---
new_docs = [
    Document(page_content="TensorFlow is a framework for building ML models.", metadata={"topic": "ml"}),
]
vectorstore.add_documents(new_docs)
print(f"\n=== Added 1 document. Total searchable documents now. ===")

# --- Search again with updated store ---
results = vectorstore.similarity_search("deep learning frameworks", k=2)
print(f"\n=== Search after adding documents ===")
for doc in results:
    print(f"  {doc.page_content}")
