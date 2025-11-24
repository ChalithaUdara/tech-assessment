# Docker Setup Guide

This guide explains how to run all services using Docker Compose.

## Prerequisites

- Docker Desktop (or Docker Engine) installed and running
- `.env` file configured with required environment variables (see [README.md](README.md#installation--setup))

## Services

The `docker-compose.yml` file includes three services:

1. **Qdrant** - Vector database (ports 6333, 6334)
2. **Chat** - Gradio chat interface (port 7860)
3. **Dashboard** - Streamlit analytics dashboard (port 8501)

## Quick Start

### 1. Create `.env` File

Ensure you have a `.env` file in the project root with required variables:

```bash
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=Gpt4o
OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
MODEL_NAME=Gpt4o

# Optional: Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=datacom_rag

# Optional: Logging Configuration
LOG_FORMAT=both
LOG_JSON_FILE=logs/chat.jsonl
```

### 2. Build and Start All Services

```bash
# Build and start all services in detached mode
docker-compose up -d

# Or build and start with logs visible
docker-compose up
```

### 3. Access Services

Once all services are running:

- **Chat Interface**: http://localhost:7860
- **Analytics Dashboard**: http://localhost:8501
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 4. Check Service Status

```bash
# View running containers
docker-compose ps

# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f chat
docker-compose logs -f dashboard
docker-compose logs -f qdrant
```

### 5. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears Qdrant data)
docker-compose down -v
```

## Service Details

### Qdrant (Vector Database)

- **Image**: `qdrant/qdrant:latest`
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Storage**: Persisted in `./qdrant_storage/`
- **Health Check**: Automatic health monitoring

### Chat Application

- **Dockerfile**: `Dockerfile.chat`
- **Port**: 7860
- **Volumes**:
  - `./logs` - Log files (shared with dashboard)
  - `./data` - Data directory for RAG documents
- **Environment**: Loads from `.env` file
- **Dependencies**: Waits for Qdrant to be healthy before starting

### Analytics Dashboard

- **Dockerfile**: `Dockerfile.dashboard`
- **Port**: 8501
- **Volumes**:
  - `./logs` - Log files (reads from `logs/chat.jsonl`)
- **Environment**: Loads from `.env` file
- **Dependencies**: Starts after Qdrant

## Building Images

To rebuild images after code changes:

```bash
# Rebuild all images
docker-compose build

# Rebuild a specific service
docker-compose build chat
docker-compose build dashboard

# Rebuild and restart
docker-compose up -d --build
```

## Troubleshooting

### Services Won't Start

1. **Check Docker is running**:
   ```bash
   docker ps
   ```

2. **Check logs for errors**:
   ```bash
   docker-compose logs
   ```

3. **Verify `.env` file exists** and contains required variables:
   ```bash
   cat .env
   ```

### Chat App Can't Connect to Qdrant

- Ensure Qdrant service is healthy: `docker-compose ps`
- Check Qdrant logs: `docker-compose logs qdrant`
- Verify `QDRANT_URL` is set to `http://qdrant:6333` (service name, not localhost)

### Dashboard Shows No Data

- Ensure chat app is generating logs
- Check that `LOG_FORMAT=both` or `LOG_FORMAT=json` in `.env`
- Verify `logs/chat.jsonl` exists and contains data
- Check dashboard logs: `docker-compose logs dashboard`

### Port Already in Use

If ports 6333, 6334, 7860, or 8501 are already in use:

1. Stop the conflicting service
2. Or modify port mappings in `docker-compose.yml`:
   ```yaml
   ports:
     - "7861:7860"  # Use 7861 on host instead
   ```

### Rebuilding After Code Changes

After modifying code, rebuild the affected service:

```bash
# Rebuild and restart chat
docker-compose up -d --build chat

# Rebuild and restart dashboard
docker-compose up -d --build dashboard
```

## Development Workflow

For development, you may prefer to run services individually:

```bash
# Start only Qdrant
docker-compose up -d qdrant

# Run chat app locally (outside Docker)
python chat.py

# Run dashboard locally (outside Docker)
streamlit run dashboard.py
```

## Production Considerations

For production deployments:

1. **Environment Variables**: Use Docker secrets or environment variable injection instead of `.env` file
2. **Resource Limits**: Add resource limits to services in `docker-compose.yml`
3. **Networking**: Use proper network isolation
4. **Logging**: Configure centralized logging
5. **Monitoring**: Add monitoring and alerting
6. **Security**: Review security best practices for containerized applications

## Network Architecture

All services run on a shared Docker network (`datacom-network`):

- Services can communicate using service names (e.g., `http://qdrant:6333`)
- External access is via published ports on `localhost`
- Internal communication doesn't require port exposure

