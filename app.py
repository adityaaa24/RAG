import streamlit as st

st.set_page_config(page_title="Samsung RAG Chatbot")
st.title("Samsung Washing Machine RAG Chatbot")

st.info("Replace this starter with your LangChain pipeline.")

question=st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write("Connect your LangChain RAG pipeline here.")
