import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate


class ChatPDF:
    def __init__(self, persist_directory="data/chroma"):
        self.persist_directory = persist_directory
        self.embedding = OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://host.docker.internal:11434"
        )
        self.llm = Ollama(
            model="mistral", base_url="http://host.docker.internal:11434"
        )

        self.prompt = PromptTemplate.from_template(
            """
            <s>[INST] 
            Eres un asistente médico experto. Responde preguntas clínicas, científicas o relacionadas con la salud basándote únicamente en el contexto proporcionado.

            Si la información no está en el contexto, responde de forma honesta: 
            "No tengo suficiente información en los documentos para responder con precisión."

            Usa lenguaje claro, preciso y profesional. Si es posible, incluye causas, síntomas o mecanismos relevantes, pero sin inventar datos.

            Mantén las respuestas detalladas, útiles y sin alucinaciones.

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
                if filename.endswith(".md") or filename.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                elif filename.endswith(".pdf"):
                    loader = PyMuPDFLoader(path)
                elif filename.endswith(".docx") or filename.endswith(".doc"):
                    loader = UnstructuredWordDocumentLoader(path)
                else:
                    continue

                docs.extend(loader.load())
                print(f"Precargado: {filename}")
            except Exception as e:
                print(f"Error en {filename}: {e}")

        if docs:
            self._split_and_store(docs)
            print(f"Total documentos precargados: {len(docs)}")

    def ingest(self, file_path, original_filename=None):
        ext = (original_filename or file_path).split(".")[-1].lower()
        if ext in ["md", "txt"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == "pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Tipo de archivo no soportado.")

        docs = loader.load()
        self._split_and_store(docs)

    def ask(self, question):
        return self.qa_chain.run(question)
