# GenAI Learning

A comprehensive portfolio of Generative AI projects outside of university assignments.

## Projects Overview

### Project 1: Exploratory Text Generation
**Skills:** Text generation, hyperparameter exploration
- Explored model: GPT-2, DistilGPT-2, facebook/opt-350m
- Generated creative text outputs

[View Project →](./projects/project-1-text-generation)

### Project 2: Semantic Search System
**Skills:** Embeddings, similarity search, data processing
- Built semantic search on 27K Steam game descriptions
- Used Sentence Transformers (all-MiniLM-L6-v2)
- Generated embeddings for all games
- Implemented cosine similarity search

[View Project →](./projects/project-2-semantic-search)

### Project 3: Containerized Semantic Search API
**Skills:** FastAPI, Docker, REST APIs, production deployment
- Converted Project 2 into production API
- Built `/search` and `/health` endpoints
- Implemented error handling & logging
- Containerized with Docker
- Can be deployed anywhere

[View Project →](./projects/project-3-semantic-search-api)

### Project 4: Fine-tuned Generative Model with Quantization
**Skills:** Model fine-tuning, quantization, benchmarking, MLOps
- Fine-tuned GPT-2 on 27K game descriptions
- Implemented INT8 quantization for efficiency
- Compared tradeoff accuracy vs efficiency

[View Project →](./projects/project-4-fine-tuning-gen-model-quantization)

### Project 5: RAG Game Recommendation System
**Skills:** RAG, LangChain, FAISS, embeddings, local open-source LLMs
- Built a game recommendation chatbot using Steam descriptions
- Uses Embedl's Hugging Face embedding model for retrieval
- Stores vectors in FAISS and retrieves relevant games
- Generates recommendations with a small local Hugging Face model

[View Project →](./projects/project-5-rag-game-recommendation)

## Learning Path

### Fundamentals
- Text generation
- Embedding models & semantic search
- Working with real datasets (27K+ records)
- REST APIs & FastAPI
- Docker containerization
- Model quantization & optimization
- Hugging Face Model Hub integration
- Retrieval-Augmented Generation (RAG)
- LangChain retrievers, vector stores, and local LLM chains

## Key Skills

✅ **ML/DL:** Transfer learning, fine-tuning, embeddings, quantization
✅ **Production:** Docker, APIs, error handling, logging
✅ **Data:** Processing, tokenization, train/validation splits
✅ **Tools:** Hugging Face, PyTorch, FastAPI, Pandas, NumPy, LangChain, FAISS

## Tech Stack

- **Languages:** Python 3.x (most in 3.14, some in 3.11)
- **ML Frameworks:** PyTorch, Transformers, Sentence Transformers, LangChain
- **Vector Search:** FAISS
- **APIs & Deployment:** FastAPI, Uvicorn, Docker
- **Data:** Pandas, NumPy, Datasets
- **MLOps:** Hugging Face Hub, Git

## Setup & Installation

Each project has its own setup instructions. See individual READMEs.
