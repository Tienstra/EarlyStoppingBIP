
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  Kernels.gaussian_kernels import GaussianKernel
from ForwardModels import models

class EKI:
    def __init__(self,forward_model, observations, dim_particles, num_particles, init_covariance, init_mean, noise_level, time):
        self.dim_observations = observations.shape[0]
        self.dim_particles = dim_particles
        self.num_particles = num_particles
        self.init_covariance = init_covariance
        self.init_mean = init_mean
        self.noise_level = noise_level
        self.forward_model = forward_model
        self.observations = observations
        self.ensemble = self._sample_prior()
        self.predictions = self._push_forward()
        self.time = time #this should be a list [start, end, num points]
    def _get_dt(self):
        return (self.time[1] - self.time[0]) / self.time[2]

    def _sample_prior(self):
        #dim particles is the number of rows and num_particles is the number of col
        #each column is a particle 
        ensemble = np.random.normal(loc=0, scale=1, size=(self.dim_particles, self.num_particles))
        return np.dot(self.init_covariance, ensemble)
    
    def generate_noise_cov(self):
        return self.noise_level*np.ones(self.dim_observations)
    
    def _push_forward(self):
        return self.forward_model.evaluate(self.ensemble)
   
    def compute_kalman_gain(self):
        #emperical covariance between theta and g_theta
        CM = np.cov(self.ensemble, self._push_forward())[self.dim_particles:(self.dim_particles + self.dim_observations), 0:self.dim_particles].T
        #emperical covariance of g_theta
        Sigma = np.cov(self._push_forward())
        #Kalman gain
        K = CM@np.linalg.inv((Sigma + (1/self._get_dt())*self.generate_noise_cov()))
        
        return K
    
    def _update_ensemble(self):
        g_bar = np.mean(self.predictions, axis=1)
        g_bar_mat = np.tile(g_bar, (self.num_particles,1)).T
        y_mat = np.tile(self.observations,(self.num_particles,1)).T
        K = self.compute_kalman_gain()
        kalman_update = K@(g_Theta + g_bar_mat - 2*y_mat)
       
        self.ensemble = self.ensemble - (1/2) * kalman_update

    def compute_residuals(self):
        return np.linalg.norm(self.observations - self.predictions)

    def fit(self):
        ts = np.arange(self.time[0], self.time[1] + self._get_dt, self._get_dt)
        for index in range(0,ts.size-1):
            #update ensemble
            self._update_ensemble()
        return self.ensemble


  
if __name__ == "__main__":
    dim_y = dim_theta = 2
    num_particles = 20
    # Initialize with proper covariance and mean
    init_mean = np.zeros(dim_theta)  # Assuming zero mean for the prior
    prior_cov = GaussianKernel(dim_theta, 2).operator_fourier
    linear_model = models.LinearForwardModel(dim_theta,num_particles,5)
    theta_dagger = np.ones(dim_theta)
    noise_level = 0.001
    time = [0,1,10]
    noise = np.random.normal(0, noise_level, dim_y)
    y = linear_model.evaluate(theta_dagger) + noise
    EKI  = EKI(linear_model, y, dim_y, num_particles, prior_cov, init_mean, noise_level, time)
    print(EKI.ensemble)

