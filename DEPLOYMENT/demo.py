

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import json

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# Load models
print("Loading models for demo...")

MODEL_DIR = Path("models/ppo_model/final")
CHROMA_DIR = Path("data/vector_store")

# LLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
model.eval()

# Embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Vector store
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection("arxiv_papers")

print("Models loaded")

# RAG pipeline
def retrieve_and_generate(query, use_rag=True, num_sources=5, temperature=0.7):
    """Complete RAG pipeline"""
    
    sources_text = ""
    
    if use_rag:
        # Retrieve
        query_embedding = embedding_model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_sources
        )
        
        # Format context
        contexts = []
        for i, (doc, metadata) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), 1):
            contexts.append(f"[{i}] {metadata['title']}\n{doc[:300]}...")
            sources_text += f"\n**Source {i}:** {metadata['title']} (arXiv:{metadata['arxiv_id']})\n"
        
        context = "\n\n".join(contexts)
        
        prompt = f"""Context from research papers:
{context}

Question: {query}

Answer:"""
    else:
        prompt = query
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean response
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    elif query in response:
        response = response.replace(query, "").strip()
    
    return response, sources_text if use_rag else "RAG disabled - no sources used"


# Gradio interface
demo = gr.Blocks(title="Multi-Agent RAG Research Assistant", theme=gr.themes.Soft())

with demo:
    gr.Markdown("""
    #  Multi-Agent RAG Research Assistant
    ### RLHF-Aligned AI for Academic Research
    
    This system combines:
    -  **10K+ ArXiv papers** in Machine Learning, AI, NLP, and Computer Vision
    -  **Semantic search** with RAG (Retrieval Augmented Generation)
    -  **RLHF alignment** for high-quality, well-cited responses
    -  **Constitutional AI** safety layer
    
    Ask questions about research papers, ML concepts, or recent advances!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="E.g., What are transformers in deep learning?",
                lines=3
            )
            
            with gr.Row():
                use_rag = gr.Checkbox(label="Use RAG (recommended)", value=True)
                num_sources = gr.Slider(1, 10, value=5, step=1, label="Number of sources")
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature")
            
            submit_btn = gr.Button("Ask", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            examples = gr.Examples(
                examples=[
                    "What are transformers in deep learning?",
                    "Explain RLHF and how it works",
                    "What is the attention mechanism?",
                    "How does LoRA fine-tuning work?",
                    "What are diffusion models?",
                    "Explain retrieval augmented generation"
                ],
                inputs=query_input,
                label="Example Questions"
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            response_output = gr.Textbox(
                label="Response",
                lines=10,
                show_copy_button=True
            )
        
        with gr.Column():
            sources_output = gr.Markdown(label="Sources")
    
    # Statistics
    gr.Markdown(f"""
    ###  System Statistics
    - **Documents in knowledge base:** {collection.count():,}
    - **Model:** GPT-2 Medium (355M params) with LoRA + RLHF
    - **Embedding model:** all-MiniLM-L6-v2
    - **Device:** {DEVICE}
    """)
    
    # Connect
    submit_btn.click(
        fn=retrieve_and_generate,
        inputs=[query_input, use_rag, num_sources, temperature],
        outputs=[response_output, sources_output]
    )

# Cell 6

print("\nLaunching Gradio demo...")
print("The interface will open in your browser")
print("Share link will be generated for remote access")

demo.launch(
    share=True,  # Creates public URL
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)


print("\n" + "="*80)
print("DEMO INTERFACE LAUNCHED")
print("="*80)
print("""
The Gradio interface is now running!

Features:
- Interactive chat interface
- RAG toggle (enable/disable retrieval)
- Adjustable number of sources
- Temperature control for generation
- Example questions
- Source citations
- Public share link (if share=True)

To stop the server, interrupt the kernel.
""")
