#  Vector Store Setup

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
import numpy as np

CHUNKS_FILE = Path("data/processed/chunks/all_chunks.json")
CHROMA_DIR = Path("data/vector_store")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

print("Vector store setup initialized")


# Load chunks
with open(CHUNKS_FILE, 'r') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully")


# Initialize ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Create collection
collection = client.get_or_create_collection(
    name="arxiv_papers",
    metadata={"description": "ArXiv research papers for RAG"}
)

print(f"ChromaDB collection created: {collection.name}")
print(f"Existing documents: {collection.count()}")


# Batch embedding and insertion
batch_size = 100
total_batches = (len(chunks) + batch_size - 1) // batch_size

print(f"\nProcessing {len(chunks)} chunks in {total_batches} batches...")

for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to ChromaDB"):
    batch = chunks[i:i+batch_size]
    
    # Prepare data
    texts = [c['text'] for c in batch]
    ids = [c['chunk_id'] for c in batch]
    metadatas = [
        {
            'arxiv_id': c['arxiv_id'],
            'title': c['title'],
            'authors': ', '.join(c['authors'][:3]),  # First 3 authors
            'published': c['published'],
            'categories': ', '.join(c['categories']),
            'chunk_index': c['chunk_index'],
        }
        for c in batch
    ]
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    
    # Add to collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

print(f"\nTotal documents in collection: {collection.count()}")

# Test retrieval
test_query = "What are transformers in deep learning?"
print(f"\nTest query: '{test_query}'")

query_embedding = model.encode([test_query])[0].tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nTop 5 results:")
for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
), 1):
    print(f"\n{i}. {metadata['title']}")
    print(f"   Distance: {distance:.4f}")
    print(f"   Preview: {doc[:150]}...")

