import argparse
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on Anthropic HH-RLHF")
    parser.add_argument("--model_name",   type=str, default="gpt2-medium")
    parser.add_argument("--output_dir",   type=str, default="models/sft_lora_weights")
    parser.add_argument("--train_samples",type=int, default=18000)
    parser.add_argument("--val_samples",  type=int, default=2000)
    parser.add_argument("--epochs",       type=int, default=3)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--grad_accum",   type=int, default=2)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--lora_alpha",   type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=100)
    return parser.parse_args()


def prepare(data):
    return Dataset.from_list([
        {"text": f"### Question: {d['chosen'].rsplit(chr(10)*2+'Assistant:', 1)[0].strip()}\n### Answer: {d['chosen'].rsplit(chr(10)*2+'Assistant:', 1)[-1].strip()}"}
        for d in data
    ])


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    raw        = load_dataset("Anthropic/hh-rlhf")
    train_data = prepare(raw["train"].select(range(args.train_samples)))
    val_data   = prepare(raw["test"].select(range(args.val_samples)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length, padding=False)

    train_dataset = train_data.map(tokenize, batched=True, remove_columns=["text"])
    val_dataset   = val_data.map(tokenize,   batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir                  = "/tmp/sft_checkpoints",
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        logging_steps               = 50,
        eval_steps                  = 500,
        save_steps                  = 500,
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
    print(f"SFT saved to {args.output_dir}")


if __name__ == "__main__":
    main()
