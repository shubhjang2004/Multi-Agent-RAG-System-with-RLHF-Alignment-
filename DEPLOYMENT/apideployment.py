


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import json


# Configuration
class Config:
    MODEL_DIR = Path("models/ppo_model/final")
    CHROMA_DIR = Path("data/vector_store")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_K = 5

config = Config()
print(f"Using device: {config.DEVICE}")

# Load models
print("Loading models...")

# LLM
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(config.DEVICE)
model.eval()

# Embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Vector store
chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
collection = chroma_client.get_collection("arxiv_papers")

print("Models loaded successfully")


# API Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_rag: Optional[bool] = True

class Source(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    relevance_score: float
    text_snippet: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Source]
    metadata: dict


# RAG functions
def retrieve_context(query: str, top_k: int = 5):
    """Retrieve relevant documents"""
    query_embedding = embedding_model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    contexts = []
    sources = []
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        contexts.append(doc)
        sources.append(Source(
            arxiv_id=metadata['arxiv_id'],
            title=metadata['title'],
            authors=metadata['authors'],
            relevance_score=1.0 - distance,  # Convert distance to similarity
            text_snippet=doc[:200]
        ))
    
    return "\n\n".join(contexts), sources

def generate_response(query: str, context: Optional[str] = None):
    """Generate response with optional RAG context"""
    if context:
        prompt = f"""Context from research papers:
{context}

Question: {query}

Answer based on the context provided:"""
    else:
        prompt = query
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part
    if "Answer based on the context provided:" in response:
        response = response.split("Answer based on the context provided:")[-1].strip()
    elif query in response:
        response = response.replace(query, "").strip()
    
    return response


# FastAPI app
app = FastAPI(
    title="Multi-Agent RAG Research Assistant",
    description="RLHF-aligned research assistant with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Multi-Agent RAG Research Assistant API",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vector_store_docs": collection.count(),
        "device": config.DEVICE
    }

@app.get("/stats")
async def get_stats():
    return {
        "total_documents": collection.count(),
        "model": "GPT-2 Medium (RLHF-aligned)",
        "vector_store": "ChromaDB",
        "embedding_model": "all-MiniLM-L6-v2"
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint with RAG"""
    try:
        sources = []
        context = None
        
        # Retrieve context if RAG is enabled
        if request.use_rag:
            context, sources = retrieve_context(request.query, request.top_k)
        
        # Generate response
        response = generate_response(request.query, context)
        
        return QueryResponse(
            query=request.query,
            response=response,
            sources=sources,
            metadata={
                "used_rag": request.use_rag,
                "num_sources": len(sources),
                "model": "gpt2-medium-ppo-aligned"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("FastAPI app created")


# Save API code to file
api_code = '''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# [Include all the code from cells 3-7 above]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

api_file = Path("api/app.py")
api_file.parent.mkdir(parents=True, exist_ok=True)

with open(api_file, 'w') as f:
    f.write(api_code)

print(f"API code saved to {api_file}")

# Cell 9
# Test the API locally
print("\nTesting API endpoints...")

test_request = QueryRequest(
    query="What are transformers in deep learning?",
    top_k=3,
    use_rag=True
)

# Simulate the endpoint
result = await query_endpoint(test_request)

print(f"\nQuery: {result.query}")
print(f"Response: {result.response}")
print(f"\nSources ({len(result.sources)}):")
for i, source in enumerate(result.sources, 1):
    print(f"{i}. {source.title}")
    print(f"   Relevance: {source.relevance_score:.4f}")

# Cell 10
# Instructions for running
print("\n" + "="*80)
print("API DEPLOYMENT INSTRUCTIONS")
print("="*80)

instructions = """
To run the API server:

1. From terminal:
   cd api
   python app.py

2. Or using uvicorn directly:
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload

3. Access the API:
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health
   - Query: POST http://localhost:8000/query

4. Example curl request:
   curl -X POST http://localhost:8000/query \\
     -H "Content-Type: application/json" \\
     -d '{"query": "What is RLHF?", "top_k": 5, "use_rag": true}'

5. Deploy to production:
   - Use Docker for containerization
   - Deploy to AWS/GCP/Azure
   - Use Nginx as reverse proxy
   - Enable HTTPS with SSL certificates
"""

print(instructions)

