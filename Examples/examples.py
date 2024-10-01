from ForwardModels import models
import numpy as np
import math 
class Example:
    def __init__(self, dim_theta, dim_y, noise_level):
        self.dim_theta = dim_theta
        self.dim_y = dim_y
        self.noise_level = noise_level


    def get_observations(self,theta_dagger):
        noise = np.random.normal(0,self.noise_level, self.dim_y)
        y = models.linear_model.evaluate(theta_dagger) + noise

class SequenceSpace(Example):
    def __init__(self, dim_theta, dim_y, beta):
        super().__init__(dim_theta, dim_y)
        self.beta = beta 

    def get_theta_dagger(self): 
        pass 


class Schroedinger(Example):
    def __init__(self, dim_theta, dim_y):
        super().__init__(dim_theta, dim_y)
       

