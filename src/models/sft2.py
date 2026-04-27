import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="Domain SFT on research preference pairs")
    parser.add_argument("--model_name",   type=str, default="gpt2-medium")
    parser.add_argument("--sft_dir",      type=str, default="models/sft_lora_weights")
    parser.add_argument("--output_dir",   type=str, default="models/sft2_lora_weights")
    parser.add_argument("--pairs_path",   type=str, default="data/rlhf/preference_data/research_preference_pairs.json")
    parser.add_argument("--train_split",  type=float, default=0.9)
    parser.add_argument("--epochs",       type=int, default=3)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--grad_accum",   type=int, default=2)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--lora_alpha",   type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.pairs_path) as f:
        pairs = json.load(f)

    split       = int(args.train_split * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base  = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if Path(args.sft_dir).exists():
        base = PeftModel.from_pretrained(base, args.sft_dir).merge_and_unload()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_config)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length, padding=False)

    train_dataset = Dataset.from_list([{"text": f"### Question: {d['question']}\n### Answer: {d['chosen']}"} for d in train_pairs]).map(tokenize, batched=True, remove_columns=["text"])
    val_dataset   = Dataset.from_list([{"text": f"### Question: {d['question']}\n### Answer: {d['chosen']}"} for d in val_pairs]).map(tokenize,   batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir                  = "/tmp/sft2_checkpoints",
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
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Domain SFT saved to {args.output_dir}")


if __name__ == "__main__":
    main()
