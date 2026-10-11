import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*Profiler clears events.*")


@pytest.fixture(autouse=True)
def _bounded_threads():
    """Keep tests light on a shared CPU and deterministic in thread count."""
    prev = torch.get_num_threads()
    torch.set_num_threads(min(4, prev))
    yield
    torch.set_num_threads(prev)


def rand_qkv(B=2, H=2, L=16, d=8, dtype=torch.float64, seed=0, Lk=None, dv=None, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    Lk = Lk or L
    dv = dv or d
    q = torch.randn(B, H, L, d, generator=g, dtype=torch.float64).to(dtype)
    k = torch.randn(B, H, Lk, d, generator=g, dtype=torch.float64).to(dtype)
    v = torch.randn(B, H, Lk, dv, generator=g, dtype=torch.float64).to(dtype)
    if requires_grad:
        q, k, v = (t.requires_grad_(True) for t in (q, k, v))
    return q, k, v
