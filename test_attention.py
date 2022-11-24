from typing import Optional, Tuple
import torch
from torch import nn
import numpy as np


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor (batch_size, length, dim)
        Input tensor.
    n_heads : int
        Number of attention heads.
    """
    batch_size, dim = x.size(0), x.size(-1)
    x = x.view(batch_size, -1, n_heads, dim // n_heads)  # (batch_size, length, n_heads, d_head)
    x = x.transpose(1, 2)  # (batch_size, n_heads, length, d_head)
    return x


# def combine_heads(x: torch.Tensor) -> torch.Tensor:
#     """
#     Parameters
#     ----------
#     x : torch.Tensor (batch_size, n_heads, length, d_head)
#         Input tensor.
#     """
#     #https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
#     batch_size, n_heads, d_head = x.size(0), x.size(1), x.size(3)
#     x = x.transpose(1, 2).contiguous().view(batch_size, -1, d_head * n_heads)  # (batch_size, length, n_heads * d_head)
#     return x


# def add_mask(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#     """
#     Mask away by setting such weights to a large negative number, so that they evaluate to 0
#     under the softmax.
#     Parameters
#     ----------
#     x : torch.Tensor (batch_size, n_heads, *, length) or (batch_size, length)
#         Input tensor.
#     mask : torch.Tensor, optional (batch_size, length)
#         Mask metrix, ``None`` if it is not needed.
#     """
#     if mask is not None:
#         if len(x.size()) == 4:
#             expanded_mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, length)
#         x = x.masked_fill(expanded_mask.bool(), -np.inf)
#     return


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Parameters
    ----------
    scale : float
        Scale factor (``sqrt(d_head)``).
    dropout : float, optional
        Dropout, ``None`` if no dropout layer.
    """
    def __init__(self, scale: float, dropout: float = 0.5) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        Q : torch.Tensor (batch_size, n_heads, length, d_head)
            Query
        K : torch.Tensor (batch_size, n_heads, length, d_head)
            Key
        V : torch.Tensor (batch_size, n_heads, length, d_head)
            Value
        mask : torch.Tensor (batch_size, 1, 1, length)
            Mask metrix, None if it is not needed
        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, length, d_head)
            Context vector.
        att : torch.Tensor (batch_size, n_heads, length, length)
            Attention weights.
        """

        q = Q / Q.norm(dim=-1, keepdim=True)
        k = K / K.norm(dim=-1, keepdim=True)

        kv = (k * V).sum(dim=-2, keepdim=True)

        att = q * kv
        att = self.dropout(att)
        # context = ...

        return att


class HydraAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int = 8, dropout: Optional[float] = None) -> None:
        super(HydraAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads

        # linear projections
        self.W_Q = nn.Linear(in_features=dim, out_features=self.d_head)
        self.W_K = nn.Linear(in_features=dim, out_features=self.d_head)
        self.W_V = nn.Linear(in_features=dim, out_features=self.d_head)

        # scaled dot-product attention
        scale = self.d_head ** 0.5  # scale factor
        self.attention = ScaledDotProductAttention(scale=scale, dropout=dropout)

        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(...)

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:

        Q = self.W_Q.forward(x)
        K = self.W_K.forward(x)
        V = self.W_V.forward(x)

        Q, K, V = split_heads(Q, self.n_heads), split_heads(K, self.n_heads), split_heads(V,
                                                                                          self.n_heads)  # (batch_size, n_heads, length, d_head)

        q = Q / Q.norm(dim=-1, keepdim=True)
        k = K / K.norm(dim=-1, keepdim=True)

        kv = (k * V).sum(dim=-2, keepdim=True)

        att = q * kv
        att = att if self.dropout is None else self.dropout(att)

        out = self.layer_norm(att)  # LayerNorm

        return out
    