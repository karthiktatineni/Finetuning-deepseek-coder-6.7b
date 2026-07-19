import os
# Force C: cache before importing transformers
os.environ["HF_HOME"] = "C:/Users/karth/.cache/huggingface"

import sys
import json
import ast
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters" / "stage1"
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

def extract_python_code(text):
    """Extract code from python markdown block."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    return text.strip()

def check_syntax(code_string):
    """Parse python code using AST to check syntax."""
    try:
        ast.parse(code_string)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e.msg} on line {e.lineno}"
    except Exception as e:
        return False, f"Error: {e}"

def generate_response(model, tokenizer, prompt):
    """Generate response using deepseek-coder chat format."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
        
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "adapter"], required=True, help="Which model to evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Evaluating Model: {args.model.upper()}")
    print("=" * 60)

    # 1. Load Tokenizer & Model (4-bit to fit in 6GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.model == "adapter":
        print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
        if not ADAPTER_DIR.exists():
            print(f"[FAIL] Adapter directory not found: {ADAPTER_DIR}")
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    # 2. Run Benchmarks
    results = {}
    total_tasks = 0
    passed_syntax = 0

    for json_file in sorted(BENCHMARK_DIR.glob("*.json")):
        category = json_file.stem
        print(f"\nCategory: {category}")
        
        with open(json_file, "r") as f:
            tasks = json.load(f)

        results[category] = []
        for task in tasks:
            print(f"  Running {task['id']}...")
            raw_output = generate_response(model, tokenizer, task["prompt"])
            
            # Code verification
            extracted_code = extract_python_code(raw_output)
            is_ok, msg = check_syntax(extracted_code)
            
            total_tasks += 1
            if is_ok:
                passed_syntax += 1

            results[category].append({
                "id": task["id"],
                "prompt": task["prompt"],
                "expected": task["expected"],
                "generated": raw_output,
                "extracted_code": extracted_code,
                "syntax_valid": is_ok,
                "syntax_message": msg
            })
            print(f"    Syntax Status: {msg}")

    # 3. Save Results
    output_file = EVAL_DIR / f"results_{args.model}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # 4. Print Summary
    print("\n" + "=" * 60)
    print("  Evaluation Complete")
    print("=" * 60)
    print(f"  Saved to: {output_file}")
    print(f"  Total Tasks evaluated: {total_tasks}")
    print(f"  Valid Syntax Code Blocks: {passed_syntax} / {total_tasks} ({passed_syntax/total_tasks:.1%})")
    print("=" * 60)

if __name__ == "__main__":
    main()
