import os
from pathlib import Path

def main():
    cache_path = Path("C:/Users/karth/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-instruct/snapshots")
    if not cache_path.exists():
        print("Cache path does not exist:", cache_path)
        return
        
    print("Listing files in snapshots:")
    for snapshot in cache_path.iterdir():
        if snapshot.is_dir():
            print("Snapshot:", snapshot.name)
            for f in snapshot.iterdir():
                is_sym = f.is_symlink()
                try:
                    target = os.readlink(f) if is_sym else "N/A"
                except Exception as e:
                    target = f"Error reading link: {e}"
                
                print(f"  {f.name} - is_symlink: {is_sym} - points to: {target}")
                
if __name__ == "__main__":
    main()
