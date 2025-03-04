# tests/test_cuda_debug.py

import pytest
import torch

def test_cuda_hello_world():
    assert torch.cuda.is_available(), "CUDA is not available"
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    z = x + y
    assert torch.all(z == torch.tensor([5.0, 7.0, 9.0], device='cuda'))