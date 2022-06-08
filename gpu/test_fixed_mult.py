import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

# Parameters
n = 64 * 1024 * 1024
N = 32
iter = 10000

# Initialize the inputs
x = torch.randint(-(1 << (N - 1)), (1 << (N - 1)), (n,), dtype=torch.int32, device='cuda')
y = torch.randint(-(1 << (N - 1)), (1 << (N - 1)), (n,), dtype=torch.int32, device='cuda')

# Profile
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    # Perform several iterations
    for i in tqdm(range(iter)):
        z = x * y

# Print profile results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
