import os
from huggingface_hub import hf_hub_download

# Force standard cache directory on C:
os.environ["HF_HOME"] = "C:/Users/karth/.cache/huggingface"

def main():
    print("Downloading missing tokenizer.model file to cache...")
    try:
        path = hf_hub_download(
            repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            filename="tokenizer.model",
            cache_dir="C:/Users/karth/.cache/huggingface"
        )
        print("Success! Downloaded to:", path)
    except Exception as e:
        print("Failed to download tokenizer.model:", e)

if __name__ == "__main__":
    main()
