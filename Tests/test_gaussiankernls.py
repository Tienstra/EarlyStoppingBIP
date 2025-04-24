import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Kernels.gaussian_kernels import GaussianKernel


def test_getOperator():
    ans = [[1.0, 0.0], [0.0, 0.25]]
    # test values of diag
    assert np.array_equal(GaussianKernel(2, 2)._operator_fourier, ans)

    # test dims
    assert GaussianKernel(2, 1)._operator_fourier.shape == (2, 2)
