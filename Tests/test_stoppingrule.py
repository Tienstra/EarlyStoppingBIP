import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EKI.stopping_rules import DiscrepancyPrinciple


def test_dp():

    residuals = 0.02
    tol = 1
    kappa = 1
    max_iter = 4
    eff_dim = 4

    assert (
        DiscrepancyPrinciple(residuals, eff_dim, tol, kappa, max_iter).converged()
        == True
    )
