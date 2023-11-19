from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.document_loaders  import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import os
import warnings
import random
import string

def  preprocess_pdf(pages):
    repo_id = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceHubEmbeddings(
                repo_id=repo_id,
                task="feature-extraction"
            )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function = len,)
    texts = text_splitter.split_documents(pages) 
    db  = FAISS.from_documents(texts, embeddings)
    return db


