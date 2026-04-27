import argparse
import torch
import chromadb
import gradio as gr
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio RAG demo")
    parser.add_argument("--model_name",  type=str,   default="gpt2-medium")
    parser.add_argument("--model_dir",   type=str,   default="models/dpo_lora_weights")
    parser.add_argument("--chroma_dir",  type=str,   default="data/chromadb")
    parser.add_argument("--collection",  type=str,   default="arxiv_papers")
    parser.add_argument("--embed_model", type=str,   default="all-MiniLM-L6-v2")
    parser.add_argument("--share",       action="store_true", default=False)
    parser.add_argument("--port",        type=int,   default=7860)
    parser.add_argument("--max_tokens",  type=int,   default=200)
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


def retrieve(query, top_k):
    emb     = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=top_k, include=["documents", "metadatas"])
    context, sources = [], []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context.append(f"[{meta['title']}]\n{doc[:300]}")
        sources.append(f"- **{meta['title']}** (`{meta['arxiv_id']}`)")
    return "\n\n".join(context), "\n".join(sources)


def generate(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = base.generate(
            **inputs,
            max_new_tokens       = args.max_tokens,
            temperature          = temperature,
            do_sample            = True,
            pad_token_id         = tokenizer.eos_token_id,
            no_repeat_ngram_size = 3,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def answer(query, use_rag, top_k, temperature):
    if use_rag:
        context, sources_md = retrieve(query, int(top_k))
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        prompt     = f"### Question: {query}\n### Answer:"
        sources_md = "RAG disabled."
    return generate(prompt, temperature), sources_md


with gr.Blocks(title="Multi-Agent RAG Research Assistant") as demo:
    gr.Markdown("# Multi-Agent RAG Research Assistant\nRLHF-aligned academic QA over ArXiv papers.")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Question", placeholder="What are transformers in deep learning?", lines=3)
            with gr.Row():
                use_rag     = gr.Checkbox(label="Use RAG", value=True)
                top_k       = gr.Slider(1, 10, value=5, step=1, label="Sources")
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature")
            submit = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    "What are transformers in deep learning?",
                    "Explain RLHF and how it works",
                    "How does LoRA fine-tuning work?",
                    "What are diffusion models?",
                    "Explain retrieval augmented generation",
                ],
                inputs=query_input,
            )

    with gr.Row():
        response_output = gr.Textbox(label="Response", lines=10, show_copy_button=True)
        sources_output  = gr.Markdown(label="Sources")

    gr.Markdown(f"**Docs indexed:** {collection.count():,} | **Device:** {device}")

    submit.click(fn=answer, inputs=[query_input, use_rag, top_k, temperature], outputs=[response_output, sources_output])


if __name__ == "__main__":
    demo.launch(share=args.share, server_port=args.port)
