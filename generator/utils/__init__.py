import torch


def get_available_gpu_memory() -> int:
    _, gpu_memory_total = torch.cuda.mem_get_info()
    return gpu_memory_total - torch.cuda.memory_allocated()
