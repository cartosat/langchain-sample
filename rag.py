# Install required libraries
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_qa_system(file_path):
    # Load PDF document
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(pages)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create QA system
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa

def ask_question(qa_system, question):
    return qa_system.run(question)

# Example usage
if __name__ == "__main__":
    # Initialize QA system with your PDF
    qa_system = create_qa_system("data/your-document.pdf")
    
    # Ask a question
    question = "What is the main topic of this document?"
    answer = ask_question(qa_system, question)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")