#  Reward Model Training


# Cell 2
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
from pathlib import Path
from tqdm import tqdm
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

DATA_DIR = Path("data/rlhf/preference_data")
MODEL_DIR = Path("models/reward_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load preference data
with open(DATA_DIR / 'train_preferences.json', 'r') as f:
    train_data = json.load(f)

with open(DATA_DIR / 'val_preferences.json', 'r') as f:
    val_data = json.load(f)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")


# Reward Model
class RewardModel(nn.Module):
    def __init__(self, base_model_name='distilbert-base-uncased'):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        reward = self.reward_head(pooled)
        return reward


# Dataset
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize chosen and rejected
        chosen_text = item['prompt'] + " " + item['chosen']
        rejected_text = item['prompt'] + " " + item['rejected']
        
        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_enc['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_enc['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(),
        }


# Initialize
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = RewardModel().to(DEVICE)

train_dataset = PreferenceDataset(train_data, tokenizer)
val_dataset = PreferenceDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)


# Training loop
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        # Get rewards for chosen and rejected
        chosen_rewards = model(
            batch['chosen_input_ids'].to(DEVICE),
            batch['chosen_attention_mask'].to(DEVICE)
        )
        
        rejected_rewards = model(
            batch['rejected_input_ids'].to(DEVICE),
            batch['rejected_attention_mask'].to(DEVICE)
        )
        
        # Loss: chosen should have higher reward
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            chosen_rewards = model(
                batch['chosen_input_ids'].to(DEVICE),
                batch['chosen_attention_mask'].to(DEVICE)
            )
            
            rejected_rewards = model(
                batch['rejected_input_ids'].to(DEVICE),
                batch['rejected_attention_mask'].to(DEVICE)
            )
            
            loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
            total_loss += loss.item()
            
            # Accuracy: chosen > rejected
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += chosen_rewards.size(0)
    
    return total_loss / len(loader), correct / total

# Train
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc = evaluate(model, val_loader)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


# Save model
torch.save(model.state_dict(), MODEL_DIR / 'reward_model.pt')
tokenizer.save_pretrained(MODEL_DIR)

print(f"\nModel saved to {MODEL_DIR}")
