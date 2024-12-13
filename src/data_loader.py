import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain  # For managing the conversation flow
from langchain.memory import ConversationBufferMemory

def extract_text_from_pdfs(pdf_paths):
    total_text = ""
    for i, pdf_path in enumerate(pdf_paths):
        print(i, "===", pdf_path, "-----------------------------------------------")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            total_text += page.extract_text() or ""  # Collecting the text from each page
    return total_text

def preprocess_chunk(chunk):
    # Remove all non-alphanumeric characters (except spaces)
    cleaned_chunk = re.sub(r'[^A-Za-z\s]', '', chunk)
    # Remove extra spaces
    cleaned_chunk = re.sub(r'\s+', ' ', cleaned_chunk).strip()

    return cleaned_chunk


# Function to split and preprocess text using RecursiveCharacterTextSplitter
def split_and_preprocess_recursively(text, chunk_size=500, chunk_overlap=50):
    # Initialize the RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators = ["\n\n", "\n"]
    )
    documents = splitter.split_text(text)
    
    return documents

def create_faiss_index(chunks, embedding_model, index_path):
    if os.path.exists(index_path):
        # If the index exists, load it
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        st.sidebar.success("Loaded existing FAISS index.")
    else:
        # If the index doesn't exist, create a new one
        vectorstore = FAISS.from_texts(chunks, embedding_model)
        vectorstore.save_local(index_path)
        st.sidebar.success("Created new FAISS index.")
    
    return vectorstore


# Define the retrieval chain function
def chain_creation(query, retriever, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
    response = conversational_chain({"question": query})
    print(response)
    return response