import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama  # Ollama integration
from langchain_core.messages import HumanMessage, AIMessage  # For structured chat history
from data_loader import extract_text_from_pdfs, split_and_preprocess_recursively, create_faiss_index, chain_creation

# App config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Question Answering Chatbot")

# Ollama model setup
embed = OllamaEmbeddings(model="nomic-embed-text")  # Use LLaMA 2 model for embeddings
llm = ChatOllama(model="llama2")  # Use LLaMA 2 model for chatbot responses

st.sidebar.header("Upload PDFs")
# Upload multiple PDFs
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)


# session state
if "chat_history" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chat_history = [AIMessage(content="Hello! How can I assist you today?")]
    
if uploaded_files and st.session_state.faiss_index is None:
    text = extract_text_from_pdfs(uploaded_files)
    # Split the text and preprocess it
    cleaned_chunks = split_and_preprocess_recursively(text, chunk_size=100, chunk_overlap=50)
    index_path = "faiss_index"
    st.session_state.faiss_index = create_faiss_index(cleaned_chunks, embed, index_path)

retriever = None
if st.session_state.faiss_index:
    retriever = st.session_state.faiss_index.as_retriever()

if retriever:
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query and st.session_state.faiss_index:
    # Add user's query to chat history as HumanMessage
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Retrieve answer from the LLaMA 2 model
        response = chain_creation(user_query, retriever, llm)  # Using the `rest` function with `ChatOllama`
    
        # Add model's response to chat history as AIMessage
        st.session_state.chat_history.append(AIMessage(content=response['answer']))

        # Display the response
        # st.markdown(f"**Answer:** {response['answer']}")
        # with st.expander("Source Documents"):
        #     for doc in response['source_documents']:
        #         st.write(doc.page_content)

    # Display the chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
