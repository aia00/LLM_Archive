import torch

def parse_dtype(dtype):
    if dtype == 'bf16':
        return torch.bfloat16
    elif dtype == 'fp32':
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]