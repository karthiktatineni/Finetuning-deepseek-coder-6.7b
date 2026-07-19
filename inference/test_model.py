import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95
# ───────────────────────────────────────────────────────────────


def load_model(model_name):
    """Load a quantized causal LM and its tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={
        0: "5GiB",
        "cpu": "28GiB"
    },
    torch_dtype=torch.float16,
)

    print("Model loaded successfully!")
    return model, tokenizer


def generate(model, tokenizer, prompt):
    """Generate a response for the given prompt."""
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model(MODEL_NAME)

    prompt = "Write a Python function to reverse a linked list."
    response = generate(model, tokenizer, prompt)
    print(response)