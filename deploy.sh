#!/bin/bash
# Deploy agent files and ChromaDB to EC2
# Usage: ./deploy.sh

EC2_HOST="ec2-user@13.223.41.3"
SSH_KEY="./rag-key.pem"
REMOTE_DIR="/home/ec2-user/app"

echo "Deploying to $EC2_HOST..."

rsync -avz --progress \
  -e "ssh -i $SSH_KEY" \
  --include="main.py" \
  --include="agent_graph.py" \
  --include="agent_stream.py" \
  --include="deps.py" \
  --include="tool.py" \
  --include="schemas.py" \
  --include="requirements.txt" \
  --include="requirements.server.txt" \
  --include="index.html" \
  --include="Dockerfile" \
  --include=".env" \
  --include="chroma_db/" \
  --include="chroma_db/**" \
  --exclude="*" \
  ./ "$EC2_HOST:$REMOTE_DIR"

echo "Building and starting Docker on EC2..."
ssh -i $SSH_KEY $EC2_HOST "cd $REMOTE_DIR && \
  docker stop rag-chatbot 2>/dev/null || true && \
  docker rm rag-chatbot 2>/dev/null || true && \
  docker build -t rag-chatbot . && \
  docker run -d -p 8000:8000 --env-file .env --name rag-chatbot rag-chatbot"

echo "Done. Server running at http://13.223.41.3:8000"
