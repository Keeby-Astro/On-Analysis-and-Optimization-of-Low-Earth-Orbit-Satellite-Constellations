import torch
import torch.nn as nn

class TriangularCausalMask(nn.Module):
    def __init__(self, max_len: int, device="cpu"):
        super().__init__()
        tri = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.register_buffer("_tri_mask", tri.to(device), persistent=False)

    def forward(self, B: int, L: int):
        m = self._tri_mask[:L, :L]
        return m.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)