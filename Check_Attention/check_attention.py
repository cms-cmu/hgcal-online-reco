import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from Mod_Center.k_center_model import GPTConfig, MultiHeadSelfAttention

cfg = GPTConfig(block_size=3333, n_layer=6, n_head=4, n_embd=64, dropout=0.1)
attn = MultiHeadSelfAttention(cfg).cuda().train()
x = torch.randn(2, 3000, 64, device="cuda", requires_grad=True)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    attn(x).sum().backward()      # forward and backward
print("Flash attention OK")