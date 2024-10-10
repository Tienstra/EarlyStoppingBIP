import numpy as np
from abc import ABC, abstractmethod

class ForwardModel(ABC):
    def __init__(self, dim_theta, dim_y):
        self.dim_theta = dim_theta
        self.dim_y = dim_y
    
    def __str__(self) -> str:
        return f'ForwardModel(dim_theta={self.dim_theta}, dim_y={self.dim_y})'
    def __repr__(self) -> str:
         return f'ForwardModel(dim_theta={self.dim_theta}, dim_y={self.dim_y})'
    @abstractmethod
    def _get_operator(self):
        pass 
    @abstractmethod
    def evaluate(self):
        pass 



class LinearForwardModel(ForwardModel): 
    def __init__(self, dim_theta, dim_y,p):
        super().__init__(dim_theta, dim_y)
        self.p = p
        self.operator = self._get_operator()

    def _get_operator(self):
        i = np.linspace(1,self.dim_theta,self.dim_theta)
        gi = np.apply_along_axis(lambda x:  x**(- self.p), 0, i)
        return np.diag(gi)
    
    def evaluate(self,theta):
        return self.operator@theta
    
    

class Schroedinger(ForwardModel): 
    pass 



#linear_model = LinearForwardModel(2,2,1)._get_operator()
#print(linear_model)