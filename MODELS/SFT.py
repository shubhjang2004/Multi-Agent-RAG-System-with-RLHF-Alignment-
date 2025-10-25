# Supervised Fine-Tuning (SFT)


import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

DATA_DIR = Path("data/rlhf/preference_data")
MODEL_DIR = Path("models/sft_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load preference data for SFT
with open(DATA_DIR / 'train_preferences.json', 'r') as f:
    train_data = json.load(f)

with open(DATA_DIR / 'val_preferences.json', 'r') as f:
    val_data = json.load(f)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")


# Prepare SFT dataset (only chosen responses)
def prepare_sft_data(data):
    sft_examples = []
    for item in data:
        text = f"### Question: {item['prompt']}\n### Answer: {item['chosen']}"
        sft_examples.append({'text': text})
    return sft_examples

train_sft = prepare_sft_data(train_data)
val_sft = prepare_sft_data(val_data)

train_dataset = Dataset.from_list(train_sft)
val_dataset = Dataset.from_list(val_sft)

print(f"SFT Train: {len(train_dataset)}, Val: {len(val_dataset)}")


# Load base model with LoRA
MODEL_NAME = "gpt2-medium"  # 355M parameters

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

print(f"Base model loaded: {MODEL_NAME}")
print(f"Parameters: {model.num_parameters():,}")


# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding=False
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

print("Datasets tokenized")


# Training arguments
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    report_to="none"
)


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("Trainer initialized")


# Train
print("\nStarting SFT training...")
trainer.train()


# Save model
trainer.save_model(MODEL_DIR / "final")
tokenizer.save_pretrained(MODEL_DIR / "final")

print(f"\nModel saved to {MODEL_DIR / 'final'}")


# Test generation
model.eval()
test_prompt = "### Question: What are transformers in deep learning?\n### Answer:"

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nTest generation:")
print(generated_text)

