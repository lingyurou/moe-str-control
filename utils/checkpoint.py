import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 如果是 DDP 模型，unwrap 之后保存原本的底层模型权重
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        'epoch': epoch,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)

def load_checkpoint(model, path, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(checkpoint['state_dict'])
        return checkpoint['epoch']
    return 0