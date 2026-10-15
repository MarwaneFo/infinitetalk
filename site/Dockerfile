# Le site seul : leger, sans GPU. Il pilote les endpoints RunPod.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py index.html ./

# Les travaux et les videos vivent ici : monter un disque persistant dessus,
# sinon tout disparait au redemarrage du conteneur.
ENV DATA_DIR=/data
VOLUME /data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
