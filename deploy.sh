#!/bin/bash
# Deploy agent files and ChromaDB to EC2
# Usage: ./deploy.sh

EC2_HOST="ec2-user@your-ec2-ip"   # change this
SSH_KEY="~/.ssh/your-key.pem"     # change this
REMOTE_DIR="/home/ec2-user/medical-rag"

echo "Deploying to $EC2_HOST..."

rsync -avz --progress \
  -e "ssh -i $SSH_KEY" \
  --include="main.py" \
  --include="agent.py" \
  --include="deps.py" \
  --include="tool.py" \
  --include="schemas.py" \
  --include="requirements.txt" \
  --include="index.html" \
  --include="chroma_db/" \
  --include="chroma_db/**" \
  --exclude="*" \
  ./ "$EC2_HOST:$REMOTE_DIR"

echo "Done. Start the server with:"
echo "  ssh -i $SSH_KEY $EC2_HOST 'cd $REMOTE_DIR && uvicorn main:app --host 0.0.0.0 --port 8000'"
