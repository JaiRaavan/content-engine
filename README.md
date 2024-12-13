# Content-Engine: Document-Based Question Answering

## Overview
Content-Engine is an AI-powered solution designed to process and understand information from multiple PDFs and provide accurate answers to user queries via a chatbot interface. The system combines document preprocessing, embedding generation, storage, and retrieval to create a seamless question-answering experience.

### **Technology Stack**
- **Frontend**: Streamlit
- **Backend**: LangChain
- **Vector Database**: FAISS

---

## **Key Functionalities**

### 1. Document Reading
The system reads and processes multiple PDFs to extract their content. This is achieved using:
- **PyPDF**: For reading PDF files and extracting textual data.
- **LangChain Document Readers**: Provides enhanced support for structured document parsing.

---

### 2. Text Splitting and Preprocessing
To ensure efficient processing, large chunks of text are split into manageable pieces and cleaned using:
- **Regex Module**: Removes unnecessary delimiters and characters for better text preprocessing.

---

### 3. Generating Embeddings
To convert the textual data into vector representations:
- **Embedding Model**: “Nomic-Embed-Text” by Ollama.
  - This model generates high-quality vector embeddings, capturing the semantic meaning of the text.

---

### 4. Storing Embeddings
Once embeddings are generated:
- **FAISS**: An efficient vector database is used to store embeddings.
  - If an index already exists, it is loaded directly to avoid redundant computations.
  - Otherwise, a new FAISS index is created.

---

### 5. Question Answering
The system answers user queries using two distinct implementations:

#### **Implementation 1: LangChain with Llama2**
- **Memory**: Uses `ConversationBufferMemory` to maintain chat history.
- **Chain**: Utilizes `ConversationalRetrievalChain` to fetch the most relevant embeddings from the FAISS index.
- **LLM**: `Llama2`, a conversational language model, generates responses grounded in the retrieved documents.

#### **Implementation 2: Hugging Face Model**
- **Embedding Model**: Sentence Transformers for generating embeddings.
- **Question Answering Pipeline**: Utilizes `deepset/roberta-base-squad2` with Hugging Face's pipeline.
  - The pipeline takes a prompt comprising the user query and context (retrieved from FAISS using similarity search).
  - Generates responses that are both context-aware and accurate.
