import json
import os
from datasets import load_from_disk
from transformers import AutoTokenizer

os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")

def main():
    print("=== Step 1: Verify Processed Dataset ===")
    ds = load_from_disk('datasets/processed/opencoder_stage2_chat')
    print("Columns:", ds.column_names)
    print("First example:")
    print(json.dumps(ds[0], indent=2)[:500] + "...\n")

    print("=== Step 2: Test Tokenization on 2 Examples ===")
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False, cache_dir="./.hf_cache")
    
    for i in range(2):
        print(f"\n--- Example {i+1} ---")
        messages = ds[i]["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            print("Chat Template Applied Text:")
            print(repr(text[:200]) + "...")
            
            token_ids = tokenizer(text)["input_ids"]
            print(f"Total tokens: {len(token_ids)}")
            print(f"First 10 token IDs: {token_ids[:10]}")
        except Exception as e:
            print(f"Tokenization failed: {e}")

if __name__ == "__main__":
    main()
