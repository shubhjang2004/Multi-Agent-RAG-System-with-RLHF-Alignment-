import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DataCollatorWithPadding
from peft import PeftModel, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="PPO alignment training")
    parser.add_argument("--model_name",    type=str,   default="gpt2-medium")
    parser.add_argument("--dpo_dir",       type=str,   default="models/dpo_lora_weights")
    parser.add_argument("--reward_dir",    type=str,   default="models/reward_model")
    parser.add_argument("--output_dir",    type=str,   default="models/ppo_lora_weights")
    parser.add_argument("--pairs_path",    type=str,   default="data/rlhf/preference_data/research_preference_pairs.json")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--mini_batch",    type=int,   default=2)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1.41e-5)
    parser.add_argument("--max_new_tokens",type=int,   default=128)
    parser.add_argument("--reward_model",  type=str,   default="distilbert-base-uncased")
    return parser.parse_args()


class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        out    = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.head(pooled).squeeze(-1)


def get_rewards(queries, responses, reward_model, reward_tokenizer, device):
    texts = [q + r for q, r in zip(queries, responses)]
    enc   = reward_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        scores = reward_model(enc["input_ids"], enc["attention_mask"])
    return [torch.tensor(s.item()) for s in scores]


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.pairs_path) as f:
        pairs = json.load(f)
    prompts = [d["question"] for d in pairs]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    if Path(args.dpo_dir).exists():
        print(f"Loading DPO weights from {args.dpo_dir}")
        base_lm = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        base_lm = PeftModel.from_pretrained(base_lm, args.dpo_dir).merge_and_unload()
        base    = AutoModelForCausalLMWithValueHead.from_pretrained(base_lm.config._name_or_path)
        base.pretrained_model.load_state_dict(base_lm.state_dict(), strict=False)

    model     = base.to(device)
    ref_model = create_reference_model(model)

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    reward_base      = AutoModel.from_pretrained(args.reward_model)
    rm               = RewardModel(reward_base)
    rm.head.load_state_dict(torch.load(Path(args.reward_dir) / "reward_head.pt", map_location="cpu"))
    rm.eval()
    rm = rm.to(device)
    for p in rm.parameters():
        p.requires_grad = False

    ppo_config = PPOConfig(
        model_name              = args.model_name,
        learning_rate           = args.lr,
        batch_size              = args.batch_size,
        mini_batch_size         = args.mini_batch,
        gradient_accumulation_steps = args.grad_accum,
        ppo_epochs              = 4,
        seed                    = 42,
        optimize_cuda_cache     = True,
    )

    ppo_trainer = PPOTrainer(
        config    = ppo_config,
        model     = model,
        ref_model = ref_model,
        tokenizer = tokenizer,
    )

    generation_kwargs = {
        "min_length":    -1,
        "top_k":         0.0,
        "top_p":         1.0,
        "do_sample":     True,
        "pad_token_id":  tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
    }

    for epoch in range(args.epochs):
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Epoch {epoch+1}"):
            batch_prompts = prompts[i:i+args.batch_size]

            query_tensors    = [tokenizer.encode(p, return_tensors="pt").squeeze().to(device) for p in batch_prompts]
            response_tensors = [ppo_trainer.generate(q, **generation_kwargs).squeeze() for q in query_tensors]
            responses        = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            rewards          = get_rewards(batch_prompts, responses, rm, reward_tokenizer, device)

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            if i % 50 == 0:
                print(f"  step {i} | reward: {np.mean([r.item() for r in rewards]):.4f} | loss: {stats.get('ppo/loss/total', 0):.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"PPO saved to {args.output_dir}")


if __name__ == "__main__":
    main()
