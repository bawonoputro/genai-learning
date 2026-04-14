import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

print("Loading data...")
df = pd.read_csv('data/steam_description_data.csv')
df['combined_description'] = df['about_the_game'] + " " + df['detailed_description']

device = "mps" if torch.backends.mps.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

descriptions = df['combined_description'].tolist()

embeddings = embedder.encode(
    descriptions,
    convert_to_tensor=True,
    show_progress_bar=True
)

np.save('embeddings.npy', embeddings.cpu().numpy())
print(f"✓ Embeddings saved to 'embeddings.npy' ({embeddings.shape})")