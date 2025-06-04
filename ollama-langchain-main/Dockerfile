FROM python:latest

# Instalar dependencias del sistema necesarias para Tesseract y procesamiento de imágenes
RUN apt-get update && apt-get install -y tesseract-ocr libsm6 libxext6 libxrender-dev

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar archivos de la aplicación
COPY requirements.txt ./
COPY app.py ./
COPY chat_pdf.py ./
COPY image_ingestion.py ./

# Instalar paquetes de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto de la aplicación
EXPOSE 8501

# Comando de inicio para ejecutar Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
