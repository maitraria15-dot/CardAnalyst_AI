## -- FAISS vector Indexing with Gemini Embeddings -- ##
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectores import FAISS

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from text chunks using Google Gemini embeddings.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001"
        google_api_key=api_key
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def save_vector_store(vector_store, save_path = "vector_db"):
    vector_store.save_local(save_path)

def load_vector_store(save_path = "vector_db"):
    api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001",
        google_api_key = api_key
    )
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization= True)


