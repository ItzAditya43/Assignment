from dataclasses import dataclass


@dataclass
class ZenithConfig:
    vocab_size: int = 8192
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2  # grouped-query attention
    hidden_dim: int = 1024  # SwiGLU intermediate size
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Trades compute for memory: recomputes each block's activations during the
    # backward pass instead of storing them, cutting peak training VRAM at the
    # cost of ~30% slower steps. Only applies during training (self.training),
    # never during KV-cached inference. Use when memory, not time, is the
    # binding constraint (e.g. a bigger model on a small local GPU).
    grad_checkpoint: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
