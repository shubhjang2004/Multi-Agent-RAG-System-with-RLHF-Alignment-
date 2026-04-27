# Multi-Agent RAG System for Academic Research

End-to-end pipeline for academic research QA over ArXiv papers, combining semantic retrieval with RLHF-aligned language models.

---

## What We Built

A multi-agent system that answers research questions by retrieving relevant chunks from a corpus of ArXiv papers and generating grounded, preference-aligned responses. The pipeline covers everything from data collection to alignment training to deployment.

**The core problem:** LLMs hallucinate citations and lack grounding when answering academic questions. We address this with RAG + preference learning.

---

## Pipeline

```
ArXiv PDFs (7,316 papers)
    ↓ chunking.py
Semantic Chunks (22,864 chunks, 512 tokens, 128 overlap)
    ↓ embedding.py
ChromaDB Vector Store (cosine similarity, all-MiniLM-L6-v2)
    ↓
                    ┌─────────────────────────┐
User Query ──────▶  │   Retrieval Agent       │  ChromaDB + MMR search
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │   Synthesis Agent       │  GPT-2 Medium + LoRA
                    │   (RLHF-aligned)        │
                    └────────────┬────────────┘
                                 ↓
                    Response + Source Citations
```

---

## Alignment Pipeline

```
GPT-2 Medium (base)
    ↓ sft.py           — SFT on 18K Anthropic HH-RLHF pairs
GPT-2 + SFT
    ↓ sft2.py          — Domain SFT on 500 research preference pairs
GPT-2 + SFT2
    ↓ dpo.py           — DPO on 500 research preference pairs
GPT-2 + DPO            ← deployed model
```

Preference pairs generated using Claude API from ArXiv chunks — domain-specific chosen/rejected pairs for academic QA.

Reward model: DistilBERT (66M, full fine-tune) trained on 400 pairs, evaluated on 100.

---

## Project Structure

```
multi_agent_rag/
├── src/
│   ├── data/
│   │   ├── data_collection.py      # Fetch ArXiv papers via API
│   │   ├── chunking.py             # PDF extraction + semantic chunking
│   │   ├── embedding.py            # Embed chunks into ChromaDB
│   │   └── preference_data.py      # Generate preference pairs via Claude API
│   ├── training/
│   │   ├── sft.py                  # SFT on HH-RLHF
│   │   ├── sft2.py                 # Domain SFT on research pairs
│   │   ├── reward_model.py         # DistilBERT reward model
│   │   ├── dpo.py                  # DPO alignment
│   │   └── ppo.py                  # PPO alignment (experimental)
│   └── inference/
│       ├── rag_pipeline.py         # Interactive RAG QA (CLI)
│       ├── evaluate.py             # Evaluation — ROUGE, BERTScore, relevance
│       ├── app.py                  # FastAPI backend
│       └── demo.py                 # Gradio frontend
├── data/
│   ├── raw/
│   │   ├── arxiv_papers/           # 7,316 PDFs (gitignored)
│   │   └── arxiv_metadata.json     # Paper metadata (gitignored)
│   ├── processed/
│   │   └── chunks/                 # 22,864 chunks across 10 batch files
│   ├── chromadb/                   # Vector store (gitignored)
│   └── rlhf/
│       └── preference_data/
│           └── research_preference_pairs.json   # 500 Claude-generated pairs
├── models/                         # All model weights (gitignored)
│   ├── sft_lora_weights/
│   ├── sft2_lora_weights/
│   ├── reward_model/
│   └── dpo_lora_weights/
├── requirements.txt
└── README.md
```

---

## Dataset

| Component | Details |
|-----------|---------|
| Papers | 7,316 ArXiv papers (cs.LG, cs.AI, cs.CL, cs.CV) — 2023 to 2025 |
| Chunks | 22,864 semantic chunks — 512 token size, 128 overlap |
| SFT data | 18,000 Anthropic HH-RLHF pairs |
| Preference pairs | 500 Claude-generated domain-specific pairs |
| Reward model train/val | 400 / 100 pairs |

---

## Models

| Model | Base | Method | Data |
|-------|------|--------|------|
| SFT | GPT-2 Medium (355M) | LoRA rank-16 | 18K HH-RLHF |
| SFT2 | GPT-2 Medium + SFT | LoRA rank-16 | 500 research pairs |
| DPO | GPT-2 Medium + SFT2 | LoRA rank-16 | 500 research pairs |
| Reward | DistilBERT (66M) | Full fine-tune | 500 research pairs |

---

## Tech Stack

- **LLM**: GPT-2 Medium + LoRA (HuggingFace Transformers, PEFT)
- **Alignment**: SFT, DPO via TRL
- **Reward Model**: DistilBERT
- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Preference Data**: Claude API (Anthropic)
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Training**: PyTorch, fp16, T4 GPU

---

## Quickstart

```bash
pip install torch transformers peft trl chromadb sentence-transformers \
            fastapi uvicorn gradio anthropic pymupdf spacy nltk rouge-score bert-score

python -m spacy download en_core_web_sm
```

### Run full pipeline

```bash
# 1. Collect data
python src/data/data_collection.py --max_papers 2500 --categories cs.LG cs.AI cs.CL cs.CV

# 2. Chunk PDFs
python src/data/chunking.py --batch_size 1000

# 3. Embed into ChromaDB
python src/data/embedding.py

# 4. Generate preference pairs
python src/data/preference_data.py --api_key YOUR_ANTHROPIC_KEY --n_pairs 500

# 5. Train SFT
python src/training/sft.py

# 6. Train reward model
python src/training/reward_model.py

# 7. Domain SFT
python src/training/sft2.py

# 8. DPO
python src/training/dpo.py

# 9. Run RAG pipeline
python src/inference/rag_pipeline.py --interactive

# 10. Evaluate
python src/inference/evaluate.py --n_samples 50
```

### API server

```bash
python src/inference/app.py
# http://localhost:8000/docs
```

### Gradio demo

```bash
python src/inference/demo.py --share
```

---

## Results

*Evaluation in progress — results will be updated.*

---

## Notes

- All model weights and data are gitignored due to size
- Colab scripts (for GPU training) are separate from VS Code scripts (for development)
- PPO training is experimental — TRL v1 API breaking changes made it unstable

---

## Author

**Shubham Jangid** — B.Tech Electrical Engineering, IIT Kanpur  
GitHub: [@shubhjang2004](https://github.com/shubhjang2004)
