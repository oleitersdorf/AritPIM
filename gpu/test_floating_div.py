import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

# Parameters
n = 64 * 1024 * 1024
Ne = 8
Nm = 23
iter = 10000

# Initialize the inputs
xs = torch.randint(low=0, high=2, size=(1, n), dtype=torch.int32, device='cuda')
xe = torch.randint(low=0, high=1 << Ne, size=(1, n), dtype=torch.int32, device='cuda')
xm = torch.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=torch.int32, device='cuda')
x = ((-1) ** xs) * (torch.ldexp((xm / (2 ** 23)), (xe - 127)).to(dtype=torch.float32))

ys = torch.randint(low=0, high=2, size=(1, n), dtype=torch.int32, device='cuda')
ye = torch.randint(low=0, high=1 << Ne, size=(1, n), dtype=torch.int32, device='cuda')
ym = torch.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=torch.int32, device='cuda')
y = ((-1) ** ys) * (torch.ldexp((ym / (2 ** 23)), (ye - 127)).to(dtype=torch.float32))

# Profile
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    # Perform several iterations
    for i in tqdm(range(iter)):
        z = x / y

# Print profile results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
