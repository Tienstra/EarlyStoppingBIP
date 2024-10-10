import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from  ForwardModels.models import LinearForwardModel

def test_linearmodel():
    ans = [[1. , 0.], [0.,0.5]]
    #test values of diag
    assert np.array_equal(LinearForwardModel(2, 2,1)._get_operator(), ans)

    #test dims 
    assert LinearForwardModel(2,2,1)._get_operator().shape == (2,2)