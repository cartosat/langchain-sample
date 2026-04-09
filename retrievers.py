# Retrievers: Fetching relevant documents for a query
# Retrievers are an interface on top of vector stores (or other sources)
# that return relevant documents. They are used in RAG (Retrieval Augmented Generation).

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Build a vector store with some knowledge ---
documents = [
    Document(page_content="LangChain was created by Harrison Chase in October 2022."),
    Document(page_content="LangChain supports Python and JavaScript."),
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe operator for chaining."),
    Document(page_content="LangSmith is a platform for debugging and monitoring LLM applications."),
    Document(page_content="LangGraph is used to build stateful, multi-actor applications with LLMs."),
]

vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# --- Create a retriever from vector store ---
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}  # return top 2 most relevant documents
)

# --- Use retriever directly ---
results = retriever.invoke("Who created LangChain?")
print("=== Retriever results ===")
for doc in results:
    print(f"  - {doc.page_content}")

