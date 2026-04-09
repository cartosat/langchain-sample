# Document Loaders: Loading data from various sources
# Document loaders read data from files, URLs, databases etc.
# and return a list of Document objects (with page_content and metadata).

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# --- TextLoader: load a plain text file ---
loader = TextLoader("data/sample.txt")
docs = loader.load()
print("=== TextLoader ===")
print(f"Number of documents: {len(docs)}")
print(f"Content: {docs[0].page_content[:100]}...")
print(f"Metadata: {docs[0].metadata}")

# --- Creating Documents manually ---
# Sometimes you need to create Document objects from your own data
manual_docs = [
    Document(page_content="Python is a programming language.", metadata={"source": "manual", "topic": "python"}),
    Document(page_content="JavaScript runs in the browser.", metadata={"source": "manual", "topic": "javascript"}),
]
print("\n=== Manual Documents ===")
for doc in manual_docs:
    print(f"  Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

# --- PyPDFLoader: load a PDF file  ---
loader = PyPDFLoader("data/sample.pdf")
pages = loader.load()
print(f"\nPDF pages: {len(pages)}")
print(f"First page: {pages[0].page_content[:200]}")

print("\n=== Document structure ===")
print("Every Document has:")
print("  - page_content: the actual text content")
print("  - metadata: a dict with source info, page numbers, etc.")
