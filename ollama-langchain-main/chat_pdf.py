import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

class ChatPDF:
    def __init__(self, persist_directory="data/chroma", model_name="llama3"):
        self.persist_directory = persist_directory

        self.embedding = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://host.docker.internal:11434"
        )

        self.llm = Ollama(
            model=model_name,
            base_url="http://host.docker.internal:11434"
        )

        self.prompt = PromptTemplate.from_template(
            """
            <s>[INST] 
        Eres un asistente médico experto. Tu función es proporcionar información médica confiable, promoviendo siempre **la consulta con un profesional de salud** antes de tomar decisiones sobre tratamientos.  

        🔹 **No fomentes la automedicación**, pero si un usuario pregunta por medicamentos, menciona opciones médicas válidas basadas en evidencia y proporciona los medicamentos o tratamientos disponibles. **Aclara siempre que un médico debe supervisar cualquier tratamiento.**  

        🔹 **Interpreta síntomas sin generar alarma.** Si un usuario describe síntomas, menciona posibles enfermedades relacionadas y **explica opciones de manejo en casa** que pueden ayudar sin sustituir una consulta médica.  

        🔹 **Aclara riesgos como la resistencia antibiótica.** Explica por qué tomar antibióticos sin indicación médica puede ser peligroso y generar resistencia bacteriana.  

        🔹 **Enseña sobre la correcta administración de medicamentos.** Explica si deben tomarse con comida, en ayunas, los horarios recomendados y precauciones.  

        🔹 **Promueve el cumplimiento de tratamientos médicos.** Explica por qué seguir una medicación según las indicaciones es fundamental para evitar recaídas y complicaciones.  

        🔹 **Si no hay suficiente información, entrega un resumen útil.** Nunca digas solo "No tengo información". En su lugar, proporciona recomendaciones generales sobre salud y prevención.  
 

        Ejemplo de respuesta responsable:  
        _"Para la enfermedad X existen opciones de tratamiento como A, B y C. Sin embargo, la automedicación puede ser riesgosa. Consulta a un profesional de la salud para determinar el mejor tratamiento según tu caso."_  

        Si el usuario proporciona síntomas, responde con:  
        - **Enfermedades relacionadas con esos síntomas**.  
        - **Formas de manejo en casa**, como hidratación, descanso, alimentación adecuada y remedios naturales seguros.  
        - **Siempre recalca que lo mejor es acudir a un médico para evaluación y diagnóstico personalizado.**  
        

Pregunta: {question}  
Contexto: {context}  
Respuesta:  
[/INST]</s>
            """
        )

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt}
        )

        self._load_pretrain_docs("pretrain_docs")

    def _split_and_store(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)

    def _load_pretrain_docs(self, folder_path):
        if not os.path.exists(folder_path):
            return

        docs = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            try:
                if filename.endswith((".md", ".txt")):
                    loader = TextLoader(path, encoding="utf-8")
                elif filename.endswith(".pdf"):
                    loader = PDFMinerLoader(path)
                elif filename.endswith((".doc", ".docx")):
                    loader = UnstructuredWordDocumentLoader(path)
                else:
                    continue

                docs.extend(loader.load())
            except Exception as e:
                print(f"Error en {filename}: {e}")

        if docs:
            self._split_and_store(docs)

    def ingest(self, file_path, original_filename=None):
        ext = (original_filename or file_path).split(".")[-1].lower()
        if ext in ["md", "txt"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == "pdf":
            loader = PDFMinerLoader(file_path)
        elif ext in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Tipo de archivo no soportado.")

        docs = loader.load()
        self._split_and_store(docs)

    def ask(self, question):
        full_response = self.qa_chain.run(question)
        
        # Filtrar solo la respuesta eliminando contexto y pregunta
        if "Respuesta:" in full_response:
            response = full_response.split("Respuesta:")[-1].strip()
        else:
            response = full_response.strip()

        return response
