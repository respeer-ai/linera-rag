# Linera RAG Service

A FastAPI service for unified document retrieval across multiple GitHub repositories using LangChain and ChromaDB with DeepSeek embeddings.

## Features

- **Unified Search**: Single endpoint to query across all indexed repositories
- **Scheduled Updates**: Automatic repository updates every 6 hours
- **Zero Downtime**: Atomic index swapping ensures continuous availability
- **DeepSeek Integration**: Uses DeepSeek embeddings for vector retrieval
- **Docker Support**: Containerized deployment

## Installation

### Local Installation
1. Clone this repository:
```bash
git clone https://github.com/your-username/linera-rag.git
cd linera-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your DeepSeek API key in `.env`:
```env
DEEPSEEK_API_KEY=your_api_key_here
```

### Docker Installation
```bash
docker build -t linera-rag .
```

## Running the Service

### Local Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Run
```bash
docker run -d -p 8000:8000 --name linera-rag \
  -e DEEPSEEK_API_KEY=your_api_key_here \
  linera-rag
```

The service will:
1. Download and index repositories on startup
2. Start a background scheduler for updates
3. Provide a query API at `http://localhost:8000/query`

## Query API

Send POST requests to `/query`:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Linera?", "top_k": 3}'
```

Response:
```json
{
  "results": [
    {
      "document": "Linera is a lightweight blockchain...",
      "metadata": {
        "source": "data/repos/linera-documentation/README.md",
        "repo": "linera-documentation",
        "chunk_index": 0,
        "total_chunks": 5
      },
      "score": 0.87
    }
  ]
}
```

## Configuration

Modify `app/config.py` for:
- Repository URLs
- Update frequency
- Chunking parameters
- Embedding model
- DeepSeek API URL

Or set environment variables when running Docker:
```bash
docker run -d -p 8000:8000 --name linera-rag \
  -e DEEPSEEK_API_KEY=your_key \
  -e DEEPSEEK_API_URL=https://your-custom-url.com \
  -e UPDATE_INTERVAL_HOURS=12 \
  -e CHUNK_SIZE=1500 \
  linera-rag
```

## Supported File Types
- Markdown (.md)
- Text (.txt)
- Rust (.rs)
- TypeScript (.ts)