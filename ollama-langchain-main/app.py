import os
import tempfile
import streamlit as st
from chat_pdf import ChatPDF

st.set_page_config(page_title="MediCS")

# Elimina tokens como [INST], <s>, etc.
def clean_response(text):
    tokens_to_remove = ["<s>", "</s>", "[INST]", "[/INST]", "[S]", "[/S]", "[s]"]
    for token in tokens_to_remove:
        text = text.replace(token, "")
    return text.strip()

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

    # 📚 Barra lateral: modelo + subida de archivo
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        # Selección de modelo
        model_name = st.selectbox("Selecciona el modelo LLM:", ["llama3", "mistral"])
        st.markdown(f"**Modelo en uso:** `{model_name}`")
        st.markdown("**Archivos de preentrenamiento:** ✅ Cargados")

        # Subida de archivo
        st.markdown("---")
        st.markdown("📂 Puedes subir archivos para seguir preguntando.")
        uploaded_files = st.file_uploader(
            "Sube un archivo (.pdf, .docx, .doc, .md)",
            type=["pdf", "docx", "doc", "md"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=read_and_save_file,
        )

    if (
        "assistant" not in st.session_state
        or st.session_state.get("current_model") != model_name
    ):
        st.session_state["assistant"] = ChatPDF(
            persist_directory="data/chroma", model_name=model_name
        )
        st.session_state["current_model"] = model_name

    # Spinner para carga
    st.session_state["ingestion_spinner"] = st.empty()

    # Mostrar historial de conversación
    display_messages()

    # Entrada de usuario
    user_input = st.chat_input("Haz tu pregunta:")
    if user_input:
        with st.spinner("Pensando..."):
            response = st.session_state["assistant"].ask(user_input)
            response = clean_response(response)

        st.session_state["messages"].append((user_input, True))
        st.session_state["messages"].append((response, False))

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    page()


    
