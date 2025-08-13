import torch
import time

# Check CUDA availability
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Install the CUDA-enabled PyTorch build and ensure an NVIDIA GPU is present.")

device = torch.device("cpu")
print(f"Using device: {device} — {torch.cuda.get_device_name(0)}")

# Optional: enable TF32 on Ampere+ GPUs for faster matmul with near-FP32 accuracy
torch.backends.cuda.matmul.allow_tf32 = True

# Matrix size (note: 15k x 15k in FP32 is ~2.7 GB for A,B,result)
matrix_size = 15000

print("Creating matrices...")
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# Warm-up
print("Warming up...")
_ = torch.mm(a, b)
torch.cuda.synchronize()

# Timed run
start_time = time.time()
print("Running matrix multiplication...")
result = torch.mm(a, b)
torch.cuda.synchronize()
end_time = time.time()

print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds.")
print(f"Result shape: {result.shape}")