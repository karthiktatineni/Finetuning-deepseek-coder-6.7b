import os
# Force standard cache on C:
os.environ["HF_HOME"] = "C:/Users/karth/.cache/huggingface"

from transformers import AutoTokenizer

def main():
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    print("Testing AutoTokenizer with use_fast=True (default)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
        print("AutoTokenizer (use_fast=True) loaded successfully!")
        print("Tokenizer type:", type(tokenizer))
    except Exception as e:
        print("AutoTokenizer (use_fast=True) failed:", e)

    print("\nTesting AutoTokenizer with default options...")
    try:
        tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("AutoTokenizer (default) loaded successfully!")
        print("Tokenizer type:", type(tokenizer2))
    except Exception as e:
        print("AutoTokenizer (default) failed:", e)

if __name__ == "__main__":
    main()
