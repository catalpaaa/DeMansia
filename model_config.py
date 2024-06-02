import torch
from dataclasses import dataclass


@dataclass
class DeMansia_tiny_config:
    img_size: tuple[int] = (224, 224)
    patch_size: tuple[int] = (16, 16)
    patch_embd_fn: str = "4_2"
    depth: int = 24
    d_model: int = 192
    num_classes: int = 1000
    learning_rate: float = 1e-3
    if_token_labeling: bool = True
    token_label_size: int = 14
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


@dataclass
class DeMansia_small_config:
    img_size: tuple[int] = (384, 384)
    patch_size: tuple[int] = (16, 16)
    patch_embd_fn: str = "4_2"
    depth: int = 24
    d_model: int = 384
    num_classes: int = 1000
    learning_rate: float = 1e-3
    if_token_labeling: bool = True
    token_label_size: int = 24
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
