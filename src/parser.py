## ***** -- Document  Ingestion & Text Extraction -- *****

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Loads a PDF document (e.g., Credit Card T&Cs) and splits it into manageable chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return chunks