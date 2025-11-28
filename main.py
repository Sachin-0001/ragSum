import streamlit as st
import shutil
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import streamlit.components.v1 as components


st.set_page_config(page_title="RAG Chatbot")
col1, col2 = st.columns(2)

with col1:
    st.button("RAG Chatbot", type="primary")
        # st.switch_page("main.py")   # reload same page

with col2:
    if st.button("Text Summarizer"):
        st.switch_page("pages/summ.py")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def session_cleanup():
    if "temp_dir" in st.session_state:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

st.session_state.on_session_end = session_cleanup


st.title("Chatbot")
st.write("Upload a document and chat with it!")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

    file_path = os.path.join(st.session_state.temp_dir, uploaded.name)

    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    st.success("File uploaded and stored temporarily ✔")

    with st.spinner("Loading PDF..."):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

    with st.spinner("Chunking..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80
        )
        chunks = splitter.split_documents(docs)

    with st.spinner("Embedding..."):
        embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    with st.spinner("Creating vector store..."):
        st.session_state.vector_db = FAISS.from_documents(chunks, embed_model)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if st.session_state.vector_db:
    user_input = st.chat_input("Ask anything about your document...")

    if user_input:


        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # Retrieval
        docs = st.session_state.vector_db.similarity_search(user_input, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful document assistant.
Answer the question using ONLY the context provided.
If the information is not present, say "The document does not contain that information."

Context:
{context}

Conversation history:
{[m for m in st.session_state.messages if m['role']=='assistant']}

User question: {user_input}
"""

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2048
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(prompt)
                answer = response.content
                st.write(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.title("⚙ Settings")

    if st.button("End Session & Cleanup"):
        session_cleanup()
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.success("All temporary files cleared and chat reset!")
