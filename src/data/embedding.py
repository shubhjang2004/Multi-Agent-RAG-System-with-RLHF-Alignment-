import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB")
    parser.add_argument("--chunks_dir",  type=str, default="data/processed/chunks")
    parser.add_argument("--chroma_dir",  type=str, default="data/chromadb")
    parser.add_argument("--collection",  type=str, default="arxiv_papers")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--reset",       action="store_true")
    return parser.parse_args()


def load_chunks(chunks_dir):
    all_chunks = []
    for path in sorted(Path(chunks_dir).glob("*.json")):
        with open(path) as f:
            all_chunks.extend(json.load(f))
    return all_chunks


def main():
    args = parse_args()

    chunks = load_chunks(args.chunks_dir)
    print(f"Loaded {len(chunks)} chunks")

    embedder = SentenceTransformer(args.embed_model)
    client   = chromadb.PersistentClient(path=args.chroma_dir)

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    try:
        collection = client.get_collection(args.collection)
        existing   = set(collection.get()["ids"])
        chunks     = [c for c in chunks if c["chunk_id"] not in existing]
        print(f"Skipping {len(existing)} already indexed | Indexing {len(chunks)} new")
    except Exception:
        collection = client.create_collection(args.collection, metadata={"hnsw:space": "cosine"})

    for i in tqdm(range(0, len(chunks), args.batch_size), desc="Embedding"):
        batch      = chunks[i:i+args.batch_size]
        texts      = [c["text"]     for c in batch]
        ids        = [c["chunk_id"] for c in batch]
        metadatas  = [
            {
                "arxiv_id":    c["arxiv_id"],
                "title":       c["title"],
                "published":   c["published"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"Done. Total indexed: {collection.count()}")


if __name__ == "__main__":
    main()