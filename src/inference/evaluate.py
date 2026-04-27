import argparse
import json
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Base vs SFT vs DPO with/without RAG")
    parser.add_argument("--base_model",  type=str, default="gpt2-medium")
    parser.add_argument("--sft_dir",     type=str, default="models/sft_lora_weights")
    parser.add_argument("--sft2_dir",    type=str, default="models/sft2_lora_weights")
    parser.add_argument("--dpo_dir",     type=str, default="models/dpo_lora_weights")
    parser.add_argument("--chroma_dir",  type=str, default="data/chromadb")
    parser.add_argument("--collection",  type=str, default="arxiv_papers")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--pairs_path",  type=str, default="data/rlhf/preference_data/research_preference_pairs.json")
    parser.add_argument("--n_samples",   type=int, default=50)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--max_tokens",  type=int, default=128)
    parser.add_argument("--output",      type=str, default="data/eval_results.json")
    return parser.parse_args()


def load_model(base_model, lora_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if lora_dir and Path(lora_dir).exists():
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


def generate(prompt, tokenizer, model, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def retrieve(query, embedder, collection, top_k):
    emb     = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=top_k, include=["documents", "metadatas"])
    return list(zip(results["documents"][0], results["metadatas"][0]))


def build_prompt(query, docs=None):
    if docs:
        context = "\n\n".join([f"[{m['title']}]\n{d[:300]}" for d, m in docs])
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return f"### Question: {query}\n### Answer:"


def rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def evaluate_model(name, tokenizer, model, samples, embedder, collection, args):
    results = []
    for sample in samples:
        query    = sample["question"]
        reference = sample["chosen"]

        pred_no_rag  = generate(build_prompt(query), tokenizer, model, args.max_tokens)
        docs         = retrieve(query, embedder, collection, args.top_k)
        pred_with_rag = generate(build_prompt(query, docs), tokenizer, model, args.max_tokens)

        results.append({
            "query":           query,
            "reference":       reference,
            "no_rag":          pred_no_rag,
            "with_rag":        pred_with_rag,
            "rouge_no_rag":    rouge(pred_no_rag, reference),
            "rouge_with_rag":  rouge(pred_with_rag, reference),
        })

    avg_no_rag   = {k: round(sum(r["rouge_no_rag"][k]   for r in results) / len(results), 4) for k in ["rouge1", "rouge2", "rougeL"]}
    avg_with_rag = {k: round(sum(r["rouge_with_rag"][k] for r in results) / len(results), 4) for k in ["rouge1", "rouge2", "rougeL"]}

    print(f"\n{name}")
    print(f"  Without RAG: {avg_no_rag}")
    print(f"  With RAG:    {avg_with_rag}")

    return {"model": name, "avg_no_rag": avg_no_rag, "avg_with_rag": avg_with_rag, "samples": results}


def main():
    args = parse_args()

    with open(args.pairs_path) as f:
        pairs = json.load(f)
    samples = pairs[:args.n_samples]

    embedder   = SentenceTransformer(args.embed_model)
    client     = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)

    models_to_eval = [
        ("Base GPT2-Medium", args.base_model, None),
        ("SFT (HH-RLHF)",   args.base_model, args.sft_dir),
        ("SFT2 (Domain)",    args.base_model, args.sft2_dir),
        ("DPO",              args.base_model, args.dpo_dir),
    ]

    all_results = []
    for name, base, lora_dir in models_to_eval:
        print(f"\nLoading {name}...")
        tokenizer, model = load_model(base, lora_dir)
        result = evaluate_model(name, tokenizer, model, samples, embedder, collection, args)
        all_results.append(result)
        del model
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
