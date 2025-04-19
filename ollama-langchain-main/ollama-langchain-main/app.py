import os
import tempfile
import streamlit as st
from chat_pdf import ChatPDF

st.set_page_config(page_title="ChatPDF")

def display_messages():
    st.subheader("Chat")
    for msg, is_user in st.session_state["messages"]:
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg)
    st.session_state["thinking_spinner"] = st.empty()

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingestando {file.name}..."):
            st.session_state["assistant"].ingest(file_path)

        os.remove(file_path)

def page():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF(persist_directory="data/chroma")

    st.header("📄 ChatPDF")

    st.subheader("Sube un documento PDF")
    st.file_uploader(
        "Sube un documento",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()

   
    user_input = st.chat_input("Haz tu pregunta:")
    if user_input:  
        with st.spinner("Pensando..."):
            response = st.session_state["assistant"].ask(user_input)
       
        st.session_state["messages"].append((user_input, True))
        st.session_state["messages"].append((response, False))
        
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    page()