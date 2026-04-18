# Semantic Search API

A Semantic search engine based on FastAPI to finds similar Steam games based on natural language queries (transformer models) 

## Features

- **Semantic Search**: Find games by describing what the user looking for from Steam database
- **Deployment Using Docker**: One command to setup

## Quick Start (Recommended)

### Using Docker

The easiest way to run the API is by Using Docker

```bash
docker pull bawonoputro/semantic-search-api:1.0
docker run -p 8000:8000 bawonoputro/semantic-search-api:1.0
```

Visit: http://localhost:8000/docs

## API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Search
```bash
POST /search
```

**Request Body (example):**
```json
{
  "query": "action rpg",
  "top_k": 2
}
```

**Response (example):**
```json
{
  "query": "action rpg",
  "results": [
    {
      "name": "Scrolls of the Lord",
      "similarity": 0.5934553146362305
    },
    {
      "name": "Brave",
      "similarity": 0.5890333652496338
    }
  ],
  "count": 2
}
```
---

## Using API through Documentation page

Once running, visit: **http://localhost:8000/docs**

This provides an interactive Swagger UI w

---

## How It Works

1. **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text to embeddings
2. **Similarity**: Computes cosine similarity between query and game descriptions
3. **Ranking**: Returns top K most similar games sorted by relevance score

---

## Technologies

- **FastAPI**: Modern, fast Python web framework
- **Sentence Transformers**: State-of-the-art text embeddings
- **Docker**: Containerization

---

## Docker Image Details

**Image:** `bawonoputro/semantic-search-api:1.0`
**Size:** ~3.1 GB
**Base:** Python 3.14-slim
**Registry:** Docker Hub

Pull the latest:
```bash
docker pull bawonoputro/semantic-search-api:1.0
```

---

## Author

[@bawonoputro](https://github.com/bawonoputro)

