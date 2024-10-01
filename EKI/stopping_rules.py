import numpy as np


class StoppingRule: 
    def __init__(self, observations, predictions, tolerance, kappa,max_iterations ):
        self.observations = observations
        self.predictions = predictions
        self.dim_predictions = np.len(predictions)
        self.tolerance = tolerance 
        self.kappa = kappa 
        self.residuals = self.compute_residuals()
        self.iteration = 0
        self.max_iterations = max_iterations
        self.converge = False


    def compute_residuals(self):
        return np.linalg.norm(self.observations - self.predictions)
    

class DiscrepencyPrinciple(StoppingRule):
    def __init__(self, observations, predictions, tolerance, kappa,max_iterations):
        super().__init__(observations, predictions,tolerance, kappa, max_iterations)

    def check_convergence(self):
         self.iteration +=1
         if self.residuals <= self.kappa*np.sqrt(self.dim_predictions*self.tolerance):
            self.converge = True

