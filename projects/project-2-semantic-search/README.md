# Semantic Search on Steam Game Database
Semantic Search project on Steam games description.

## Overview
This project uses sentence transformers and semantic similarity to search Steam games by their description. Instead of using keyword matching, user can describe what they want and find relevant games.

**Example:**

Query: coop survival
Results:
1. Project Winter (Score: 0.521)
2. How to Survive 2 (Score: 0.515)
3. TRAPPED (Score: 0.504)

Query: rougelike
Results:
1. Streets of Rogue (Score: 0.400)
2. Dead Cells (Score: 0.389)
3. Refill your Roguelike (Score: 0.378)


## Running the application

### 1. Clone the repository

### 2. Download data files
Download steam_description_data.csv and steam.csv from [Kaggle Steam Store Games](https://www.kaggle.com/datasets/nikdavis/steam-store-games)
Both of them have combined size of ~200 MB

Place both files in the data/ folder

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


### 4.Generate embeddings
```bash
python generate_embeddings.py
```
This will create embeddings.npy from steam_description_data.csv

### 5. Start the application
```bash
python search.py
```

Then enter your query:
```bash
Enter search query (or 'quit' to exit): action rpg
Number of results (default 5): 3

Searching for: 'action rpg'
============================================================
1. Scrolls of the Lord (Score: 0.593)
2. Brave (Score: 0.589)
3. Fearless Fantasy (Score: 0.588)
============================================================
```

## How It Works
### 1. Embeddings Generation (generate_embeddings.py)
- Loads all game descriptions from steam_description_data.csv
- Uses sentence-transformers/all-MiniLM-L6-v2 model from SBERT
- Converts each description to a 384-dimensional vector
- Saves embeddings to embeddings.npy

### 2.Semantic Search (search.py)
- Takes user natural language query
- Converts it to an embedding using the same model
- Computes cosine similarity with all game embeddings
- Returns top-k games ranked by similarity score

## Requirements

Python 3.8+
numpy
pandas
torch
sentence-transformers

See requirements.txt for exact versions.

## License

This project uses the [Steam dataset from Kaggle](https://www.kaggle.com/datasets/nikdavis/steam-store-games). Please refer to the dataset's license for usage rights.

## Author
Paripurna Bawonoputro | 
github: bawonoputro
