import os
from transformers import LlamaTokenizer, AutoTokenizer

os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")

def main():
    print("Loading LlamaTokenizer...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
        print("LlamaTokenizer loaded successfully!")
    except Exception as e:
        print("LlamaTokenizer failed:", e)

    print("\nLoading AutoTokenizer...")
    try:
        tokenizer2 = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
        print("AutoTokenizer loaded successfully!")
    except Exception as e:
        print("AutoTokenizer failed:", e)

if __name__ == "__main__":
    main()
