import os
import re
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage  # For structured chat history
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings

# App config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Question Answering Chatbot")


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embedding_function = HuggingFaceEmbeddings(model_name=model_name)

st.sidebar.header("Upload PDFs")
# Upload multiple PDFs
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)


def load_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        print(loader.load())
        documents.extend(loader.load())  # Extract text from the PDF
    return documents


def preprocess_chunk(chunk):
    # Remove all non-alphanumeric characters (except spaces)
    cleaned_chunk = re.sub(r'[^A-Za-z\s]', '', chunk)
    # Remove extra spaces
    cleaned_chunk = re.sub(r'\s+', ' ', cleaned_chunk).strip()

    return cleaned_chunk

# Function to split and preprocess text using RecursiveCharacterTextSplitter
def split_and_preprocess_recursively(docs, chunk_size=500, chunk_overlap=50):
    # Initialize the RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split and preprocess each document
    processed_docs = []
    for doc in docs:
        # Apply the chunk splitter to each document's content
        chunks = splitter.split_text(doc.page_content)

        # Preprocess each chunk and preserve the original metadata (e.g., source, page number)
        for chunk in chunks:
            cleaned_chunk = preprocess_chunk(chunk)
            processed_docs.append(Document(page_content=cleaned_chunk, metadata=doc.metadata))  # Keep metadata intact

    return processed_docs


def store_in_faiss(texts, embedding_model, index_path):
    # Initialize a FAISS index
    if os.path.exists(index_path):
        # If the index exists, load it
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        st.sidebar.success("Loaded existing FAISS index.")
    else:
        # If the index doesn't exist, create a new one
        vectorstore = FAISS.from_documents(texts, embedding_model)
        vectorstore.save_local(index_path)
        st.sidebar.success("Created new FAISS index.")
    
    return vectorstore


def query_faiss(vectorstore, query):
    search_results = vectorstore.similarity_search(query)
    return search_results


pipe = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)


# session state
if "chat_history" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chat_history = [AIMessage(content="Hello! How can I assist you today?")]


if uploaded_files and st.session_state.faiss_index is None:
    filepaths = [f"../data/{file.name}" for file in uploaded_files]
    text = load_pdfs(filepaths)
    # Split the text and preprocess it
    cleaned_chunks = split_and_preprocess_recursively(text, chunk_size=100, chunk_overlap=50)

    index_path = "hf_faiss_index"
    st.session_state.faiss_index = store_in_faiss(cleaned_chunks, embedding_function, index_path)


vectorstore = None
if st.session_state.faiss_index:
    vectorstore = st.session_state.faiss_index

if vectorstore:
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query and st.session_state.faiss_index:
    # Add user's query to chat history as HumanMessage
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        results = query_faiss(vectorstore, user_query)
        # Retrieve answer from the LLaMA 2 model
        context = ""
        for doc in results:
            context += doc.page_content
        # Structure the prompt
        prompt = {
            "question": user_query,
            "context": context
        }
        response = pipe(prompt)
    
        # Add model's response to chat history as AIMessage
        st.session_state.chat_history.append(AIMessage(content=response['answer']))


    # Display the chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)