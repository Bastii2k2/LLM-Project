import os
import tempfile
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ImageProcessor:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def ingest_image(self, image_file):
        """Carga una imagen, extrae el texto y lo guarda en la base de datos de vectores."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(image_file.getbuffer())
            file_path = tf.name
        
        try:
            text = self.extract_text(file_path)
            if text:
                self.store_text(text)
                print(f"Texto extraído y almacenado desde {image_file.name}.")
            else:
                print(f"No se pudo extraer texto de {image_file.name}.")
        except Exception as e:
            print(f"Error procesando {image_file.name}: {e}")
        finally:
            os.remove(file_path)

    def extract_text(self, image_path):
        """Extrae texto desde una imagen usando OCR (Pytesseract)."""
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()

    def store_text(self, text):
        """Divide el texto y lo almacena en el vectorstore para su recuperación."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        self.vectorstore.add_texts(chunks)
