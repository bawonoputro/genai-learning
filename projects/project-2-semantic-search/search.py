import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import argparse

# Load data
df_descriptions = pd.read_csv('data/steam_description_data.csv')
df_games = pd.read_csv('data/steam.csv')

# Lookup dictionary
game_lookup = dict(zip(df_games['appid'], df_games['name']))

# Load model and embeddings
device = "mps" if torch.backends.mps.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
corpus_embeddings = torch.from_numpy(np.load('embeddings.npy')).to(device)

def semantic_search(query, top_k=5):
    """Search for games using natural language query"""
    query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    
    formatted_res = []
    for rank, result in enumerate(results[0], 1):
        corpus_id = result['corpus_id']
        score = result['score']
        
        steam_appid = df_descriptions.iloc[corpus_id]['steam_appid']
        game_name = game_lookup.get(steam_appid, "Unknown Game")
        
        formatted_res.append({
            'rank': rank,
            'game': game_name,
            'score': score
        })
    
    return formatted_res

if __name__ == "__main__":
    while True:
        print("\n" + "=" * 60)
        query = input("Enter search query (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        #checking if it's just space
        if not query.strip():
            print("Please enter a valid query")
            continue
        
        top_k = input("Number of results (default 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5
        
        print(f"\nSearching for: '{query}' in the database")
        print("=" * 60)
        
        results = semantic_search(query, top_k=top_k)
        
        for r in results:
            print(f"{r['rank']}. {r['game']} (Score: {r['score']:.3f})")
        
        print("=" * 60)