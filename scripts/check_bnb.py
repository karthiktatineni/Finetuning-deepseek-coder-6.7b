import bitsandbytes as bnb
import torch

print("bitsandbytes:", bnb.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())