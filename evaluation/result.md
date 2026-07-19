# LLM Fine-Tuning Evaluation: Base Model vs. Stage 1 Adapter

This document summarizes the performance comparison of the base model `deepseek-ai/deepseek-coder-6.7b-instruct` against the trained Stage 1 LoRA adapter on code syntax validation across various benchmarks.

## Summary Table

| Category | Base Syntax Pass Rate | Stage 1 Adapter Syntax Pass Rate | Improvement |
| --- | --- | --- | --- |
| Algorithms | 4/5 (80.0%) | 5/5 (100.0%) | +20.0% |
| Debugging | 3/5 (60.0%) | 3/5 (60.0%) | +0.0% |
| Fastapi | 4/5 (80.0%) | 4/5 (80.0%) | +0.0% |
| Python | 5/5 (100.0%) | 5/5 (100.0%) | +0.0% |
| React | 0/5 (0.0%) | 0/5 (0.0%) | +0.0% |
| Sql | 0/5 (0.0%) | 0/5 (0.0%) | +0.0% |
| **TOTAL** | **16/30 (53.3%)** | **17/30 (56.7%)** | **+3.3%** |

## Detailed Analysis
The evaluation measures **syntax validity** of generated code blocks using AST parsing (`ast.parse`).
- **Base Model**: Evaluates the pre-trained `deepseek-ai/deepseek-coder-6.7b-instruct` model without modifications.
- **Stage 1 Adapter**: Evaluates the model fine-tuned using QLoRA on the OpenCoder Stage 2 dataset.