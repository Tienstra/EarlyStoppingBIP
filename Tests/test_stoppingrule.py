import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EKI.stopping_rules import DiscrepancyPrinciple

def test_dp(): 
    residuals = np.logspace(-1, -5, 5)
    tol = 1e-3
    kappa = 1
    max_iter = 5

    assert DiscrepancyPrinciple(residuals, tol, kappa, max_iter) == True 
    
