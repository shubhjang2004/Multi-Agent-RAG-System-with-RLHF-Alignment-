import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward model on research preference pairs")
    parser.add_argument("--model_name",   type=str, default="distilbert-base-uncased")
    parser.add_argument("--pairs_path",   type=str, default="data/rlhf/preference_data/research_preference_pairs.json")
    parser.add_argument("--output_dir",   type=str, default="models/reward_model")
    parser.add_argument("--train_split",  type=float, default=0.8)
    parser.add_argument("--epochs",       type=int, default=5)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--grad_accum",   type=int, default=2)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=20)
    return parser.parse_args()


class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out    = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.head(pooled).squeeze(-1)


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen   = model(inputs["chosen_input_ids"],   inputs["chosen_attention_mask"])
        rejected = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"])
        loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
        return (loss, {"chosen": chosen, "rejected": rejected}) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        self.model.eval()
        correct = total = 0
        for batch in self.get_eval_dataloader():
            batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
            with torch.no_grad():
                c = self.model(batch["chosen_input_ids"],   batch["chosen_attention_mask"])
                r = self.model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            correct += (c > r).sum().item()
            total   += len(c)
        acc = correct / total
        print(f"Preference Accuracy: {acc*100:.2f}%")
        return {"eval_accuracy": acc}


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.pairs_path) as f:
        pairs = json.load(f)

    split       = int(args.train_split * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base      = AutoModel.from_pretrained(args.model_name)
    model     = RewardModel(base)

    def make_dataset(data):
        chosen_enc   = tokenizer([d["question"] + " " + d["chosen"]   for d in data], truncation=True, max_length=args.max_length, padding="max_length")
        rejected_enc = tokenizer([d["question"] + " " + d["rejected"] for d in data], truncation=True, max_length=args.max_length, padding="max_length")
        ds = Dataset.from_dict({
            "chosen_input_ids":        chosen_enc["input_ids"],
            "chosen_attention_mask":   chosen_enc["attention_mask"],
            "rejected_input_ids":      rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        })
        ds.set_format("torch")
        return ds

    train_dataset = make_dataset(train_pairs)
    val_dataset   = make_dataset(val_pairs)

    training_args = TrainingArguments(
        output_dir                  = "/tmp/reward_checkpoints",
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        logging_steps               = 10,
        eval_steps                  = 50,
        save_steps                  = 50,
        eval_strategy               = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_accuracy",
        greater_is_better           = True,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    trainer = RewardTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
    )

    trainer.train()
    base.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.head.state_dict(), Path(args.output_dir) / "reward_head.pt")
    print(f"Reward model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
