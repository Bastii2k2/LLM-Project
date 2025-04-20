import os
import tempfile
import streamlit as st
from chat_pdf import ChatPDF

st.set_page_config(page_title="MediCS")

def display_messages():
    st.subheader("Chat")
    for msg, is_user in st.session_state["messages"]:
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg)
    st.session_state["thinking_spinner"] = st.empty()

def read_and_save_file():
    uploaded_files = st.session_state.get("uploaded_files", [])
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingestando {file.name}..."):
            st.session_state["assistant"].ingest(file_path, file.name)

        os.remove(file_path)
        st.session_state["messages"].append((f"📂 Documento {file.name} cargado.", False))

def page():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("🩺 MediCS - Tu Asistente Médico Inteligente")

    model_name = st.selectbox("Selecciona el modelo LLM:", ["llama3", "mistral"])

    if (
        "assistant" not in st.session_state
        or st.session_state.get("current_model") != model_name
    ):
        st.session_state["assistant"] = ChatPDF(
            persist_directory="data/chroma", model_name=model_name
        )
        st.session_state["current_model"] = model_name

    # Estado del modelo actual
    with st.sidebar:
        st.markdown("### ⚙️ Estado")
        st.markdown(f"**Modelo en uso:** `{st.session_state['current_model']}`")
        st.markdown("**Archivos de preentrenamiento:** ✅ Cargados")
        st.markdown("---")
        st.markdown("Puedes subir archivos PDF, Word o Markdown para seguir preguntando.")

    uploaded_files = st.file_uploader(
        "📂 Sube un archivo (.pdf, .docx, .doc, .md, .txt)",
        type=["pdf", "docx", "doc", "md", "txt"],
        accept_multiple_files=True,
        key="uploaded_files",
        on_change=read_and_save_file,
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
    