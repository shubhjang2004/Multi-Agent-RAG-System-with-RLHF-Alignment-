# Preference Data Generation

import json
import random
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

OUTPUT_DIR = Path("data/rlhf/preference_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Preference data generation setup")


# Load chunks for context
with open('data/processed/chunks/all_chunks.json', 'r') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# Sample contexts
contexts = random.sample(chunks, min(1000, len(chunks)))
print(f"Sampled {len(contexts)} contexts for preference generation")


# Generate questions from contexts
def generate_question_from_context(context_text):
    """Generate research questions from paper chunks"""
    # Simple template-based generation
    templates = [
        f"What does this paper say about {context_text.split()[0]}?",
        "Can you explain the main concept discussed here?",
        "What are the key findings in this research?",
        "How does this work compare to prior research?",
        "What methodology is used in this study?"
    ]
    return random.choice(templates)

# Generate question-context pairs
qa_pairs = []
for context in tqdm(contexts[:500], desc="Generating QA pairs"):
    question = generate_question_from_context(context['text'])
    qa_pairs.append({
        'question': question,
        'context': context['text'][:500],  # Limit context length
        'arxiv_id': context['arxiv_id'],
        'title': context['title']
    })

print(f"Generated {len(qa_pairs)} QA pairs")


# Create preference pairs (good vs bad responses)
preference_data = []

for qa in tqdm(qa_pairs, desc="Creating preference pairs"):
    # Good response: accurate, well-cited, helpful
    good_response = f"Based on the research paper '{qa['title']}', {qa['context'][:200]}... This demonstrates the key findings from the study."
    
    # Bad response: vague, no citations, potentially inaccurate
    bad_responses = [
        "I'm not sure about that.",
        "This is a complex topic that requires more research.",
        "The paper discusses some interesting ideas but I can't recall specifics.",
    ]
    bad_response = random.choice(bad_responses)
    
    preference_data.append({
        'prompt': qa['question'],
        'chosen': good_response,
        'rejected': bad_response,
        'context': qa['context'],
        'source': qa['arxiv_id']
    })

print(f"Created {len(preference_data)} preference pairs")


# Save preference data
train_split = int(0.8 * len(preference_data))
train_data = preference_data[:train_split]
val_data = preference_data[train_split:]

with open(OUTPUT_DIR / 'train_preferences.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open(OUTPUT_DIR / 'val_preferences.json', 'w') as f:
    json.dump(val_data, f, indent=2)

print(f"\nSaved preference data:")
print(f"  Train: {len(train_data)} pairs")
print(f"  Val: {len(val_data)} pairs")
