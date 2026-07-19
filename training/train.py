"""Main entry point for LoRA fine-tuning."""

from config import TrainingConfig
from trainer import load_quantized_model, load_tokenizer, load_datasets, create_trainer


def main():
    config = TrainingConfig()

    print("=" * 60)
    print("  Qwen2.5-Coder LoRA Fine-Tuning")
    print("=" * 60)

    tokenizer = load_tokenizer(config)
    model = load_quantized_model(config)
    train_ds, val_ds = load_datasets(config)

    if train_ds is None:
        print("ERROR: No training data found. Run the data pipeline first.")
        return

    trainer = create_trainer(model, tokenizer, train_ds, val_ds, config)

    print("\nStarting training...")
    trainer.train()

    # Save final adapter
    adapter_path = f"{config.output_dir}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
