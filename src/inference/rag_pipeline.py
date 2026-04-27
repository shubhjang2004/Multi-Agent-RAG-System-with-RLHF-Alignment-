import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG Pipeline for Academic Research")
    parser.add_argument("--chroma_dir",   type=str, default="data/chromadb")
    parser.add_argument("--model_dir",    type=str, default="models/dpo_lora_weights")
    parser.add_argument("--base_model",   type=str, default="gpt2-medium")
    parser.add_argument("--embed_model",  type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--collection",   type=str, default="arxiv_papers")
    parser.add_argument("--top_k",        type=int, default=5)
    parser.add_argument("--fetch_k",      type=int, default=20)
    parser.add_argument("--max_tokens",   type=int, default=256)
    parser.add_argument("--query",        type=str, default=None)
    parser.add_argument("--interactive",  action="store_true", default=True)
    return parser.parse_args()


def load_retriever(chroma_dir, collection, embed_model):
    embedder  = SentenceTransformer(embed_model)
    client    = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection)
    return embedder, collection


def retrieve(query, embedder, collection, top_k=5, fetch_k=20):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    scored = sorted(zip(docs, metadatas, distances), key=lambda x: x[2])

    seen, unique = set(), []
    for doc, meta, dist in scored:
        if meta["arxiv_id"] not in seen:
            seen.add(meta["arxiv_id"])
            unique.append((doc, meta, dist))
        if len(unique) == top_k:
            break

    return unique


def build_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"[{meta['title']} ({meta['arxiv_id']})]\n{doc[:400]}"
        for doc, meta, _ in retrieved_docs
    ])
    return (
        f"You are a research assistant. Use the following paper excerpts to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def load_model(base_model, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if Path(model_dir).exists():
        model = PeftModel.from_pretrained(base, model_dir)
        model = model.merge_and_unload()
    else:
        model = base

    model.eval()
    return tokenizer, model


def generate(prompt, tokenizer, model, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def format_citations(retrieved_docs):
    return [
        {"arxiv_id": meta["arxiv_id"], "title": meta["title"], "published": meta.get("published", "")}
        for _, meta, _ in retrieved_docs
    ]


def answer_query(query, embedder, collection, tokenizer, model, args):
    retrieved = retrieve(query, embedder, collection, top_k=args.top_k, fetch_k=args.fetch_k)
    prompt    = build_prompt(query, retrieved)
    response  = generate(prompt, tokenizer, model, max_tokens=args.max_tokens)
    citations = format_citations(retrieved)
    return {"query": query, "answer": response, "citations": citations}


def main():
    args = parse_args()

    print("Loading retriever...")
    embedder, collection = load_retriever(args.chroma_dir, args.collection, args.embed_model)

    print("Loading model...")
    tokenizer, model = load_model(args.base_model, args.model_dir)

    if args.query:
        result = answer_query(args.query, embedder, collection, tokenizer, model, args)
        print(f"\nAnswer: {result['answer']}")
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  [{c['arxiv_id']}] {c['title']}")
        return

    if args.interactive:
        print("\nRAG Pipeline ready. Type 'quit' to exit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = answer_query(query, embedder, collection, tokenizer, model, args)
            print(f"\nAnswer:\n{result['answer']}")
            print("\nCitations:")
            for c in result["citations"]:
                print(f"  [{c['arxiv_id']}] {c['title']}")
            print()


if __name__ == "__main__":
    main()
