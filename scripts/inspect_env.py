import os

def main():
    print("=== HF Environment Variables ===")
    for k, v in os.environ.items():
        if "HF" in k or "TRANSFORMERS" in k or "CACHE" in k:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
