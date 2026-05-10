FROM python:3.13-slim

WORKDIR /app

# install system deps needed by some langchain/docling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.server.txt .
RUN pip install --no-cache-dir -r requirements.server.txt

# copy app source
COPY main.py agent.py deps.py tool.py schemas.py index.html ./

# copy the pre-built vectorstore
COPY chroma_db/ ./chroma_db/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
