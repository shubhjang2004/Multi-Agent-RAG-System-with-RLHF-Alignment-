import argparse
import json
import random
from pathlib import Path

import anthropic
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate preference pairs using Claude API")
    parser.add_argument("--chunks_path", type=str, default="data/processed/chunks/all_chunks.json")
    parser.add_argument("--output_path", type=str, default="data/rlhf/preference_data/research_preference_pairs.json")
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--n_pairs",     type=int, default=500)
    parser.add_argument("--model",       type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--max_tokens",  type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def generate_pair(chunk, client, model, max_tokens):
    prompt = (
        f"Paper: {chunk['title']}\n"
        f"Chunk: {chunk['text'][:800]}\n\n"
        f"Generate a question answerable from this chunk. Then write:\n"
        f"1. A GOOD answer: specific, grounded in the paper, technically accurate\n"
        f"2. A BAD answer: vague, generic, not grounded in the paper content\n\n"
        f"Return ONLY valid JSON:\n"
        '{{"question": "...", "chosen": "...", "rejected": "..."}}'
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def main():
    args = parse_args()
    random.seed(args.seed)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.chunks_path) as f:
        chunks = json.load(f)

    sampled = random.sample(chunks, min(args.n_pairs, len(chunks)))
    client  = anthropic.Anthropic(api_key=args.api_key)

    pairs  = []
    failed = 0

    for chunk in tqdm(sampled, desc="Generating pairs"):
        try:
            pair = generate_pair(chunk, client, args.model, args.max_tokens)
            pair["arxiv_id"] = chunk["arxiv_id"]
            pair["title"]    = chunk["title"]
            pairs.append(pair)
        except Exception:
            failed += 1

    with open(args.output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated: {len(pairs)} | Failed: {failed} | Saved: {args.output_path}")


if __name__ == "__main__":
    main()