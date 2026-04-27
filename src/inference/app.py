import argparse
import torch
import chromadb
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI RAG server")
    parser.add_argument("--model_name",   type=str, default="gpt2-medium")
    parser.add_argument("--model_dir",    type=str, default="models/dpo_lora_weights")
    parser.add_argument("--chroma_dir",   type=str, default="data/chromadb")
    parser.add_argument("--collection",   type=str, default="arxiv_papers")
    parser.add_argument("--embed_model",  type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--host",         type=str, default="0.0.0.0")
    parser.add_argument("--port",         type=int, default=8000)
    parser.add_argument("--max_tokens",   type=int, default=150)
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--top_k",        type=int, default=5)
    return parser.parse_args()


args   = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
if Path(args.model_dir).exists():
    base = PeftModel.from_pretrained(base, args.model_dir).merge_and_unload()
base.eval()

embedder   = SentenceTransformer(args.embed_model)
client     = chromadb.PersistentClient(path=args.chroma_dir)
collection = client.get_collection(args.collection)


class QueryRequest(BaseModel):
    query:   str
    top_k:   Optional[int]  = 5
    use_rag: Optional[bool] = True


class Source(BaseModel):
    arxiv_id:        str
    title:           str
    relevance_score: float
    snippet:         str


class QueryResponse(BaseModel):
    query:    str
    response: str
    sources:  List[Source]
    used_rag: bool


def retrieve(query, top_k):
    emb     = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    sources = []
    context = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        context.append(f"[{meta['title']}]\n{doc[:400]}")
        sources.append(Source(
            arxiv_id        = meta["arxiv_id"],
            title           = meta["title"],
            relevance_score = round(1.0 - dist, 4),
            snippet         = doc[:200],
        ))
    return "\n\n".join(context), sources


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = base.generate(
            **inputs,
            max_new_tokens      = args.max_tokens,
            temperature         = args.temperature,
            do_sample           = True,
            pad_token_id        = tokenizer.eos_token_id,
            no_repeat_ngram_size= 3,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


app = FastAPI(title="Multi-Agent RAG Research Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "device": device, "docs_indexed": collection.count()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        sources = []
        context = None
        if request.use_rag:
            context, sources = retrieve(request.query, request.top_k or args.top_k)
        prompt   = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:" if context else f"### Question: {request.query}\n### Answer:"
        response = generate(prompt)
        return QueryResponse(query=request.query, response=response, sources=sources, used_rag=request.use_rag)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
