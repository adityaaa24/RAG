import os
import streamlit as st

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Samsung Washing Machine Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Samsung Washing Machine Assistant")
st.markdown("Ask anything about your Samsung washing machine manual.")

# -------------------------
# API Key
# -------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("❌ OpenAI API Key not found.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------
# Manual Location
# -------------------------
HTML_FILE = "How to use the various modes of the washing machine | Samsung LEVANT.html"

if not os.path.exists(HTML_FILE):
    st.error(
        "Manual not found.\n\n"
        "Create a folder named data and put samsung_manual.html inside it."
    )
    st.stop()

# -------------------------
# Load HTML
# -------------------------
@st.cache_resource
def load_documents():

    loader = UnstructuredHTMLLoader(HTML_FILE)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)

documents = load_documents()

# -------------------------
# Embeddings
# -------------------------
@st.cache_resource
def create_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return db

vectorstore = create_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k":4
    }
)
# -------------------------
# LLM
# -------------------------

@st.cache_resource
def load_llm():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


qa = load_llm()


# -------------------------
# Chat History
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# Ask Question
# -------------------------

question = st.chat_input(
    "Ask your question about Samsung washing machine..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching manual..."):

            try:

                result = qa.invoke(
                    {
                        "query": question
                    }
                )

                answer = result["result"]

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                with st.expander("📄 Source Chunks"):

                    for i, doc in enumerate(result["source_documents"]):

                        st.write(f"### Source {i+1}")

                        st.write(doc.page_content)

                        st.divider()

            except Exception as e:

                st.error(str(e))
