import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# -------------------- Core RAG Logic --------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

def get_embeddings(text_chunks):
    embeddings = CohereEmbeddings(
        cohere_api_key=COHERE_API_KEY,
        user_agent="my-rag-chatbot"
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_conversational_chain():
    llm = Cohere(cohere_api_key=COHERE_API_KEY)
    return load_qa_chain(llm=llm, chain_type="stuff")

def handle_user_input(user_question, vectorstore, chain):
    docs = vectorstore.similarity_search(user_question)
    response = chain.run(input_documents=docs, question=user_question)
    return response


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        padding: 0.6rem 1rem;
        border-radius: 18px;
        margin: 10px 0;
        display: inline-block;
        float: right;
        max-width: 80%;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        padding: 0.6rem 1rem;
        border-radius: 18px;
        margin: 10px 0;
        display: inline-block;
        float: left;
        max-width: 80%;
    }
    .stChatInput > div > div {
        background-color: #f0f8ff !important;
        border-radius: 1rem;
        padding: 0.5rem;
    }
    .stChatInput input {
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    .stChatInput button {
        align-self: center !important;
        margin-top: 0.25rem !important;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        background: #ffffff;
        text-align: center;
        font-size: 0.85rem;
        color: #888;
        padding: 0.5rem 0;
        z-index: 9999;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="footer">
        © 2025 Vamsi Krishna Devatha. All rights reserved.
    </div>
""", unsafe_allow_html=True)



st.title("📄 Chat with Your PDFs")

# -------------------- State Management --------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Each item: {"role": "user"/"assistant", "content": "..."}

# -------------------- Sidebar Upload --------------------

uploaded_files = st.sidebar.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("Submit and Process"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF before processing.")
    else:
        raw_text = get_pdf_text(uploaded_files)
        chunks = get_text_chunks(raw_text)
        st.session_state.vectorstore = get_embeddings(chunks)
        st.session_state.chain = get_conversational_chain()
        st.session_state.chat_history = []  # Reset chat history on new docs
        st.success("Documents processed successfully. You can now ask questions.")

# -------------------- Chat History Display --------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='{bubble_class}'>{message['content']}</div>", unsafe_allow_html=True)

# -------------------- Chat Input & Logic --------------------

if st.session_state.vectorstore and st.session_state.chain:
    user_question = st.chat_input("Type your question")
    if user_question:
        # Add user question to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Get response and add to history
        response = handle_user_input(user_question, st.session_state.vectorstore, st.session_state.chain)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Re-render new message pair immediately
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{user_question}</div>", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            st.markdown(f"<div class='bot-bubble'>{response}</div>", unsafe_allow_html=True)

elif not uploaded_files:
    st.info("Please upload PDFs and click 'Submit and Process' to start chatting.")
