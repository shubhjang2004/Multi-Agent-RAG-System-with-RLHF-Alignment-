# Notebook 09: PPO Alignment Training
# Cell 1
!pip install transformers trl torch accelerate peft -q

# Cell 2
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from pathlib import Path
from tqdm import tqdm
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

SFT_MODEL_DIR = Path("models/sft_model/final")
REWARD_MODEL_DIR = Path("models/reward_model")
OUTPUT_DIR = Path("models/ppo_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell 3
# Load SFT model
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Wrap with value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
model = model.to(DEVICE)

print("SFT model loaded")

# Cell 4
# Load reward model
from torch import nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name='distilbert-base-uncased'):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(pooled)
        return reward

reward_model = RewardModel()
reward_model.load_state_dict(torch.load(REWARD_MODEL_DIR / 'reward_model.pt', map_location=DEVICE))
reward_model = reward_model.to(DEVICE)
reward_model.eval()

reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_DIR)

print("Reward model loaded")

# Cell 5
# Load training prompts
with open('data/rlhf/preference_data/train_preferences.json', 'r') as f:
    train_data = json.load(f)

prompts = [item['prompt'] for item in train_data[:500]]  # Limit for faster training
print(f"Loaded {len(prompts)} training prompts")

# Cell 6
# PPO Config
ppo_config = PPOConfig(
    model_name="gpt2-medium",
    learning_rate=1.41e-5,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,
    ppo_epochs=4,
    seed=42,
    steps=500,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
)

# Cell 7
# PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=prompts,
)

print("PPO Trainer initialized")

# Cell 8
# Generation kwargs
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

# Cell 9
# Reward function
def get_reward(query_response_pairs):
    """Compute rewards using reward model"""
    texts = [q + r for q, r in query_response_pairs]
    
    encodings = reward_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(DEVICE)
    
    with torch.no_grad():
        rewards = reward_model(
            encodings['input_ids'],
            encodings['attention_mask']
        )
    
    return rewards.squeeze().cpu().tolist()

# Cell 10
# Training loop
print("\nStarting PPO training...")
output_length_sampler = LengthSampler(64, 128)

stats_history = []

for epoch in range(3):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/3")
    print(f"{'='*60}")
    
    for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch['input_ids']
        
        # Generate responses
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze())
        
        batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Compute rewards
        query_response_pairs = [
            (tokenizer.decode(q), r) 
            for q, r in zip(query_tensors, batch['response'])
        ]
        
        rewards = get_reward(query_response_pairs)
        rewards = [torch.tensor(r) for r in rewards]
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        stats_history.append(stats)
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx}")
            print(f"  Mean reward: {np.mean([r.item() for r in rewards]):.4f}")
            if 'ppo/loss/total' in stats:
                print(f"  PPO loss: {stats['ppo/loss/total']:.4f}")

# Cell 11
# Save aligned model
model.save_pretrained(OUTPUT_DIR / "final")
tokenizer.save_pretrained(OUTPUT_DIR / "final")

print(f"\nAligned model saved to {OUTPUT_DIR / 'final'}")

# Cell 12
# Test aligned model
model.eval()
test_prompts = [
    "What are transformers in NLP?",
    "Explain RLHF in simple terms",
    "How does attention mechanism work?"
]

print("\n" + "="*60)
print("Testing aligned model:")
print("="*60)

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {generated}")
    print("-" * 60)

print("\nNotebook 09 Complete!")