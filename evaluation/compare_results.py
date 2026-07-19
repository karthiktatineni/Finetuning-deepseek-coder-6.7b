import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

def load_results(path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    base_results = load_results(EVAL_DIR / "results_base.json")
    adapter_results = load_results(EVAL_DIR / "results_adapter.json")

    if not base_results:
        print("Base model results not found. Please run: python evaluation/run_eval.py --model base")
        return

    markdown_lines = []
    markdown_lines.append("# LLM Fine-Tuning Evaluation: Base Model vs. Stage 1 Adapter\n")
    markdown_lines.append("This document summarizes the performance comparison of the base model `deepseek-ai/deepseek-coder-6.7b-instruct` against the trained Stage 1 LoRA adapter on code syntax validation across various benchmarks.\n")
    markdown_lines.append("## Summary Table\n")
    markdown_lines.append("| Category | Base Syntax Pass Rate | Stage 1 Adapter Syntax Pass Rate | Improvement |")
    markdown_lines.append("| --- | --- | --- | --- |")

    categories = sorted(base_results.keys())
    total_base_passed = 0
    total_adapter_passed = 0
    total_tasks = 0

    chart_data = {
        "categories": [],
        "base_scores": [],
        "adapter_scores": []
    }

    for cat in categories:
        base_tasks = base_results[cat]
        adapter_tasks = adapter_results[cat] if adapter_results else None

        base_passed = sum(1 for t in base_tasks if t["syntax_valid"])
        cat_total = len(base_tasks)
        total_tasks += cat_total
        total_base_passed += base_passed

        base_rate = base_passed / cat_total
        chart_data["categories"].append(cat.capitalize())
        chart_data["base_scores"].append(round(base_rate * 100, 1))
        
        if adapter_tasks:
            adapter_passed = sum(1 for t in adapter_tasks if t["syntax_valid"])
            total_adapter_passed += adapter_passed
            adapter_rate = adapter_passed / cat_total
            improvement = f"{adapter_rate - base_rate:+.1%}"
            adapter_str = f"{adapter_passed}/{cat_total} ({adapter_rate:.1%})"
            chart_data["adapter_scores"].append(round(adapter_rate * 100, 1))
        else:
            adapter_str = "N/A (Not Evaluated)"
            improvement = "N/A"
            chart_data["adapter_scores"].append(0)

        markdown_lines.append(f"| {cat.capitalize()} | {base_passed}/{cat_total} ({base_rate:.1%}) | {adapter_str} | {improvement} |")

    # Totals
    base_total_rate = total_base_passed / total_tasks
    if adapter_results:
        adapter_total_rate = total_adapter_passed / total_tasks
        total_improvement = f"{adapter_total_rate - base_total_rate:+.1%}"
        adapter_total_str = f"{total_adapter_passed}/{total_tasks} ({adapter_total_rate:.1%})"
    else:
        adapter_total_str = "N/A"
        total_improvement = "N/A"

    markdown_lines.append(f"| **TOTAL** | **{total_base_passed}/{total_tasks} ({base_total_rate:.1%})** | **{adapter_total_str}** | **{total_improvement}** |")

    # Add detail explanation placeholder
    markdown_lines.append("\n## Detailed Analysis")
    markdown_lines.append("The evaluation measures **syntax validity** of generated code blocks using AST parsing (`ast.parse`).")
    markdown_lines.append("- **Base Model**: Evaluates the pre-trained `deepseek-ai/deepseek-coder-6.7b-instruct` model without modifications.")
    markdown_lines.append("- **Stage 1 Adapter**: Evaluates the model fine-tuned using QLoRA on the OpenCoder Stage 2 dataset.")
    
    # Save markdown file
    output_path = EVAL_DIR / "result.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
        
    print(f"[OK] Generated {output_path}")

    # Output raw chart data in console so we can visualize it
    print("\n[CHART_DATA]")
    print(json.dumps(chart_data))

if __name__ == "__main__":
    main()
