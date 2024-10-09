import numpy as np

class GaussianKernel: 
    def __init__(self, dim_theta, dim_y, alpha):
        self._dim_theta = dim_theta
        self._dim_y = dim_y
        self._alpha = alpha
        self._operator_fourier = self._get_operator()

    @property
    def alpha(self):
        return self._alpha
    
    @property
    def dim_theta(self):
        return self._dim_theta
    
    @property
    def dim_y(self):
        return self._dim_y
   
    @dim_theta.setter
    def dim_theta(self, new_dim_theta):
        self._dim_y = new_dim_theta
    
    @dim_y.setter
    def dim_y(self, new_dim_y):
        self._dim_y = new_dim_y
    
    @alpha.setter
    def alpha(self, new_alpha):
        self._alpha = new_alpha
        self._operator_fourier = self._get_operator()


    def _get_operator(self):
        i = np.linspace(1,self._dim_theta,self._dim_theta)
        ci = np.apply_along_axis(lambda x:  x**(- self.alpha), 0, i)
        return np.diag(ci)
    
    def __str__(self) -> str:
        return f'PriorCovariance(dim_theta={self._dim_theta}, dim_y={self._dim_y})'
    def __repr__(self) -> str:
         return f'PriorCovariance(dim_theta={self._dim_theta}, dim_y={self._dim_y})'
    
if __name__ == "__main__":
    prior_cov = GaussianKernel(2,2,2)
    prior_cov.alpha = 0.5
    print(prior_cov)
    print(prior_cov._operator_fourier)
#print(__name__)