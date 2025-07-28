import torch
import torch.distributed as dist
import os

def init_process():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])  # Ensure each process gets a unique GPU

    # Set device for each process
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        torch.cuda.set_device(local_rank)  # Ensure each rank uses a different GPU
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}/{world_size} initialized successfully on {device}.")

    # Ensure tensor is on the correct device
    tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    target = (world_size * (world_size - 1)) // 2
    print(f"Rank {rank} received reduced tensor: {tensor.item()}, target: {target}")

    # barrier
    dist.barrier()
    print(f"Rank {rank} passed the barrier.")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    init_process()
    