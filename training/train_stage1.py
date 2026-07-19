import os
# Force standard cache on C:
os.environ["HF_HOME"] = "C:/Users/karth/.cache/huggingface"

from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Removed SFTTrainer since dataset is already tokenized

# Configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tokenized" / "opencoder_stage2_tokenized"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "adapters" / "stage1"

def main():
    print("=" * 60)
    print("  Stage 1 Training (QLoRA)")
    print("=" * 60)

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset
    print(f"Loading tokenized dataset from {DATASET_PATH}...")
    dataset = load_from_disk(str(DATASET_PATH))
    # SMOKE TEST: Subset the dataset
    subset_size = min(100, len(dataset))
    print(f"SMOKE TEST: Subsetting to {subset_size} examples...")
    dataset = dataset.select(range(subset_size))
    print(f"Dataset size: {len(dataset)}")

    # 3. Load Model (4-bit QLoRA)
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,      # VRAM optimization
        gradient_accumulation_steps=16,     # Effective batch size = 16
        gradient_checkpointing=True,        # CRITICAL for 6GB VRAM
        optim="paged_adamw_8bit",           # Less VRAM than 32bit
        save_steps=20,                      # Save often for smoke test
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=20,                       # SMOKE TEST limit
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False
    )

    # 6. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Start Training
    print("Starting training...")
    trainer.train()

    # 8. Save Adapter
    print(f"Saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done!")

if __name__ == "__main__":
    main()
