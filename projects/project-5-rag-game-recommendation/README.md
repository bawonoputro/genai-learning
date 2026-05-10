# RAG Game Recommendation System

A Retrieval-Augmented Generation (RAG) project that recommends Steam games based on natural language queries.

## Overview

This project builds on the Steam game dataset from Project 2. Instead of only returning semantically similar games, this project retrieves relevant game descriptions with FAISS and uses a small local language model to generate recommendation text.

**Example:**

Query: story-driven RPG

Results:
1. Retrieve similar Steam games using Embedl embeddings
2. Pass the retrieved games into a LangChain RAG chain
3. Generate a short recommendation response using a local Hugging Face model

## Features

- **Game Recommendation RAG**: Retrieve relevant Steam games and generate recommendations
- **Real Embedl Embeddings**: Default embedding model is `embedl/all-MiniLM-L6-v2-quantized-trt` from Hugging Face
- **LangChain Chain**: Uses an LCEL-style chain built with `RunnableLambda` when LangChain is installed
- **FAISS Vector Search**: Stores game description embeddings for fast similarity search
- **Local LLM**: Uses `distilgpt2` so no hosted LLM API key is required
- **CLI Demo**: Run recommendations directly from the terminal

## Running the Application

### 1. Clone the repository

```bash
git clone git@github.com:bawonoputro/genai-learning.git
cd genai-learning/projects/project-5-rag-game-recommendation
```

### 2. Prepare data

This project reuses the Steam dataset from Project 2.

Make sure these files exist:

```text
../project-2-semantic-search/data/steam_description_data.csv
../project-2-semantic-search/data/steam.csv
```

If the files are missing, download them from [Kaggle Steam Store Games](https://www.kaggle.com/datasets/nikdavis/steam-store-games) and place them in the Project 2 `data/` folder.

### 3. Create environment and install dependencies

Recommended Python: **3.11+**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Use the Embedl model from Hugging Face

The default Embedl model is public on Hugging Face, so you usually do **not** need to accept gated model terms or login first.

Model page:

```text
https://huggingface.co/embedl/all-MiniLM-L6-v2-quantized-trt
```

Optional: login if you hit anonymous download rate limits:

```bash
huggingface-cli login
```

If you only want a quick fallback experiment without the Embedl PT2 artifact, pass:

```bash
--embedding-model sentence-transformers/all-MiniLM-L6-v2
```

### 5. Build the FAISS index

For a quick first run:

```bash
python game_recommender.py "I want a story-driven RPG" --rebuild-index --limit 1000 --no-llm
```

For the full Steam dataset:

```bash
python game_recommender.py "I want a story-driven RPG" --rebuild-index --no-llm
```

This creates a local vector index in:

```text
vector_index/steam_faiss/
```

### 6. Start the demo

```bash
python game_recommender.py "I want a horror survival game"
```

Retrieval-only mode:

```bash
python game_recommender.py "Give me a multiplayer shooter" --no-llm
```

Other example queries:

```bash
python game_recommender.py "I want a story-driven RPG"
python game_recommender.py "Relaxing puzzle game"
python game_recommender.py "Give me a multiplayer shooter"
```

## Sample Outputs

These are example outputs from retrieval-only mode. Exact games can change depending on the dataset version and index size.

### Query 1: story-driven RPG

```bash
python game_recommender.py "I want a story-driven RPG" --no-llm
```

```text
Retrieved context:
1. Story Kingdom
   Steam app id: 30
   Description: A story-driven fantasy RPG. Make choices, explore kingdoms, and follow a cinematic narrative.

Recommendation:
Recommendations for: I want a story-driven RPG
1. Story Kingdom - This matches because its Steam description is relevant to your request.
```

### Query 2: multiplayer shooter

```bash
python game_recommender.py "Give me a multiplayer shooter" --no-llm
```

```text
Retrieved context:
1. Co-op Blaster
   Steam app id: 10
   Description: A co-op tactical shooter. Team up with friends for fast online battles.

Recommendation:
Recommendations for: Give me a multiplayer shooter
1. Co-op Blaster - This matches because its Steam description is relevant to your request.
```

### Query 3: relaxing puzzle game

```bash
python game_recommender.py "Relaxing puzzle game" --no-llm
```

```text
Retrieved context:
1. Puzzle Lake
   Steam app id: 20
   Description: A relaxing puzzle adventure. Solve calm logic puzzles around a peaceful lake.

Recommendation:
Recommendations for: Relaxing puzzle game
1. Puzzle Lake - This matches because its Steam description is relevant to your request.
```

## How It Works

### 1. Data Preparation (`data.py`)

- Loads `steam_description_data.csv` and `steam.csv`
- Cleans HTML tags, missing values, and extra spaces
- Combines each game title with its description
- Creates game documents for retrieval

### 2. Embedl Embeddings (`embeddings.py`)

- Default model: `embedl/all-MiniLM-L6-v2-quantized-trt`
- Downloads the Embedl `.pt2` artifact from Hugging Face
- Uses the upstream tokenizer from `sentence-transformers/all-MiniLM-L6-v2`
- Produces normalized sentence embeddings for game descriptions and queries

### 3. Vector Store (`vector_store.py`)

- Stores embeddings in FAISS
- Loads an existing index if available
- Creates a retriever for top-k similarity search

### 4. RAG Chain (`rag.py`)

- Retrieves relevant games
- Formats retrieved games as context
- Builds a recommendation prompt
- Composes the retrieval and generation steps as a LangChain Runnable chain

Prompt template:

```text
You are a game recommender.
Use the context to recommend 3 games that best match the user's request.
```

### 5. Generator (`generator.py`)

- Uses Hugging Face `distilgpt2`
- Wraps the local model with LangChain
- Generates recommendation text from the retrieved context

## Project Structure

```text
project-5-rag-game-recommendation/
├── README.md
├── requirements.txt
├── game_recommender.py
├── data/
│   └── sample/
├── src/
│   └── game_rag/
│       ├── data.py
│       ├── embeddings.py
│       ├── generator.py
│       ├── rag.py
│       └── vector_store.py
└── tests/
    └── test_game_rag_core.py
```

## Requirements

Python 3.11+

Main libraries:

- langchain
- langchain-community
- langchain-core
- langchain-huggingface
- faiss-cpu
- huggingface-hub
- sentence-transformers
- transformers
- torch
- pandas
- numpy

See `requirements.txt` for exact dependency list.

## Testing

Run tests:

```bash
python -m unittest discover -s tests -v
```

Current test coverage includes:

- Text cleaning
- Steam CSV loading
- Game document formatting
- Recommendation prompt construction
- Embedl model default configuration
- LangChain-style RAG chain construction

## Key Learnings

1. **RAG combines retrieval and generation**: the retriever finds relevant games, and the LLM turns them into recommendations.
2. **Embedl models can be used from Hugging Face**: this project uses Embedl's quantized MiniLM artifact as the default embedding model.
3. **Vector databases improve search**: FAISS makes similarity search fast after embeddings are created.
4. **LangChain connects the workflow**: the project uses documents, vector stores, retrievers, prompts, and a Runnable chain.
5. **Small local LLMs are easy to run**: `distilgpt2` works without hosted LLM API keys, although quality is limited compared to instruction-tuned models.
6. **Project 2 becomes reusable infrastructure**: the semantic search dataset can be extended into a full RAG application.

## Notes

- The generated `vector_index/` folder is not committed to GitHub.
- Full dataset indexing may take time because every Steam description needs an embedding.
- The default Embedl model is public on Hugging Face. Login is optional, but can help if you hit anonymous rate limits.
- `distilgpt2` is used for simplicity. A future improvement would be replacing it with a stronger instruction-tuned local model.

## Author

Paripurna Bawonoputro  
GitHub: bawonoputro
