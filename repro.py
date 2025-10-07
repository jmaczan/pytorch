import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class MixedFakeModeModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = torch.nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Process input first - this creates fake tensors in export's fake mode
        processed = self.lin(x)

        # Create some computation that depends on processed tensor
        intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

        def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
            threshold = intermediate[
                batch_idx, q_idx % seq_len
            ]  # Access the captured tensor
            return (kv_idx <= q_idx) & (threshold > 0)

        block_mask = create_block_mask(
            mask_mod=dynamic_mask_function,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=x.device,
            _compile=False,  # HF sets this to True, which runs into the issue i am talking below
        )
        q = processed.view(batch_size, 1, seq_len, self.dim)
        k = processed.view(batch_size, 1, seq_len, self.dim)
        v = processed.view(batch_size, 1, seq_len, self.dim)

        # this doesn't work
        out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
        # this works (flex attention internally calls torch.compile(backend=eager) which
        # has special handling similar to torch.cond
        out = flex_attention(q, k, v, block_mask=block_mask)

        return out

torch.compile(MixedFakeModeModel(), fullgraph=True)(torch.randn(2, 128, 64))
