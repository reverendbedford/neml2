import torch
import neml2
from neml2.tensors import *


class TestBatchTensor:
    def test_constructors(self):
        """Constructors"""

        A = torch.arange(60).reshape(2, 3, 5, 2)
        B = BatchTensor(A, 2)
        assert torch.allclose(A, B.to_torch())
