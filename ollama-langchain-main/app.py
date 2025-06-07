import os
import tempfile
import streamlit as st
from chat_pdf import ChatPDF
from image_ingestion import ImageProcessor

st.set_page_config(page_title="🩺 MediCS - Tu Asistente Médico Inteligente")

def clean_response(text):
    tokens_to_remove = ["<s>", "</s>", "[INST]", "[/INST]", "[S]", "[/S]", "[s]", "</INST>","[/RESP]"]
    for token in tokens_to_remove:
        text = text.replace(token, "")
    return text.strip()

def display_messages():
    """Muestra el historial de conversación."""
    st.subheader("Chat")
    for msg, is_user in st.session_state["messages"]:
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg)
    st.session_state["thinking_spinner"] = st.empty()

def process_uploaded_files():
    """Procesa archivos subidos sin repetir mensajes anteriores."""
    uploaded_files = st.session_state.get("uploaded_files", [])
    
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()  # Guardará archivos ya procesados

    nuevos_mensajes = []

    for file in uploaded_files:
        if file.name in st.session_state["processed_files"]:
            continue  # Evita procesar archivos duplicados

        ext = file.name.split(".")[-1].lower()
        st.session_state["processed_files"].add(file.name)  # Registra el archivo procesado

        if ext in ["png", "jpg", "jpeg"]:
            with st.spinner(f"Procesando imagen {file.name}..."):
                st.session_state["image_processor"].ingest_image(file)
            nuevos_mensajes.append((f"🖼️ Imagen {file.name} procesada y almacenada.", False))

        elif ext in ["pdf", "docx", "doc", "md", "txt"]:
            with st.spinner(f"Ingestando documento {file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(file.getbuffer())
                    file_path = tf.name
                st.session_state["assistant"].ingest(file_path, file.name)
                os.remove(file_path)
            nuevos_mensajes.append((f"📂 Documento {file.name} cargado.", False))

    # Agregar solo los nuevos mensajes, evitando duplicación
    st.session_state["messages"].extend(nuevos_mensajes)


def page():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("🩺 MediCS - Tu Asistente Médico Inteligente")

    # 📚 Barra lateral: modelo + subida de archivos
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        model_name = st.selectbox("Selecciona el modelo LLM:", ["llama3", "mistral"])
        st.markdown(f"**Modelo en uso:** `{model_name}`")
        st.markdown("**Archivos de preentrenamiento:** ✅ Cargados")

        # Subida de archivos (documentos e imágenes en un solo uploader)
        st.markdown("---")
        st.markdown("📂 Puedes subir documentos e imágenes para análisis.")

        uploaded_files = st.file_uploader(
            "Sube archivos (.pdf, .docx, .doc, .md, .txt, .png, .jpg, .jpeg)",
            type=["pdf", "docx", "doc", "md", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=process_uploaded_files,
        )

    if "assistant" not in st.session_state or st.session_state.get("current_model") != model_name:
        st.session_state["assistant"] = ChatPDF(
            persist_directory="data/chroma", model_name=model_name
        )
        st.session_state["image_processor"] = ImageProcessor(st.session_state["assistant"].vectorstore)
        st.session_state["current_model"] = model_name

    # Spinner de carga
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
