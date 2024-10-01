import numpy as np


class StoppingRule: 
    def __init__(self, oberservations, predictions, tolerance, kappa):
        self.oberservations = oberservations
        self.predictions = predictions
        self.dim_predictions = np.len(predictions)
        self.tolerance = tolerance 
        self.kappa
        self.rediduals = self.compute_residulas()

    def compute_residulas(self):
        return np.linalg.norm(self.oberservations - self.predictions)
    

class DiscrepencyPrinciple(StoppingRule):
    def __init__(self, oberservations, predictions,tolerance, kappa):
        super().__init__(oberservations, predictions,tolerance, kappa)

    def stop(self):
         if self.rediduals <= self.kappa*np.sqrt(self.dim_predictions*self.tolerance):
            False