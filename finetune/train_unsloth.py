#!/usr/bin/env python3
"""
Fine-tune a model using Unsloth (4x faster, 70% less memory).

Prerequisites:
    uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    uv pip install --no-deps trl peft accelerate bitsandbytes

Usage:
    python train_unsloth.py --base-model unsloth/Qwen2.5-7B-bnb-4bit --epochs 3
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with Unsloth")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-7B-bnb-4bit",
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--output-dir", default="./terraform-expert-lora")
    args = parser.parse_args()

    # Import here to fail fast if not installed
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "training" / "alpaca.json"
    print(f"Loading dataset: {data_path}")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Format prompt
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, inp, output in zip(instructions, inputs, outputs):
            if inp:
                text = f"""### Instruction:
{instruction}

### Input:
{inp}

### Response:
{output}"""
            else:
                text = f"""### Instruction:
{instruction}

### Response:
{output}"""
            texts.append(text + tokenizer.eos_token)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=args.output_dir,
            save_strategy="epoch",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete: {trainer_stats}")

    # Save LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    # Export to GGUF for Ollama
    print("Exporting to GGUF format...")
    model.save_pretrained_gguf(
        args.output_dir + "-gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(f"GGUF model saved to {args.output_dir}-gguf/")


if __name__ == "__main__":
    main()
