# Imagen para Hugging Face Spaces (SDK Docker, CPU gratuita).
# Los modelos se ejecutan dentro del contenedor con transformers: sin token ni créditos.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Spaces ejecuta el contenedor con el usuario 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    INFERENCE_BACKEND=local
WORKDIR /home/user/app

# 1. Dependencias (torch en su versión CPU, mucho más ligera).
COPY --chown=user requirements.txt requirements-local.txt ./
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements-local.txt

# 2. Modelos precargados en la imagen: el primer análisis no tiene que descargarlos.
COPY --chown=user core/__init__.py core/config.py core/hf_client.py core/local_inference.py core/
COPY --chown=user scripts/prefetch_models.py scripts/
RUN python scripts/prefetch_models.py

# 3. Código de la aplicación.
COPY --chown=user . .

EXPOSE 8501
# XSRF/CORS desactivados: Spaces sirve la app dentro de un iframe y, si no, la subida de archivos falla.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
