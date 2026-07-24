import torch


def select_torch_device(device: str = "auto") -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    return device