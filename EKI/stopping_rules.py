import numpy as np
from abc import ABC, abstractmethod


class StoppingRule(ABC): 
    def __init__(self, residuals, effective_dim, tolerance, kappa,max_iterations):
        self.residuals = residuals
        self.effective_dim = effective_dim
        self.tolerance = tolerance 
        self.kappa = kappa 
        self.iteration = 0
        self.max_iterations = max_iterations
        self.converge = False
    @abstractmethod
    def check_convergence():
        pass 
    

class DiscrepancyPrinciple(StoppingRule):
    def __init__(self,residuals, tolerance, kappa,max_iterations):
        super().__init__(residuals, tolerance, kappa,max_iterations)

    def check_convergence(self):
        self.iteration +=1
        if self.residuals <= self.kappa*(np.sqrt(self.effective_dim*self.tolerance)):
            print("Converged")
            self.converge = True
        elif self.residuals > self.kappa*np.sqrt(self.dim_predictions*self.tolerance):
            print("Reached Max Iterations")
            self.converge = False
    

