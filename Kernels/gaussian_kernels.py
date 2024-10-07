import numpy as np

class GaussianKernel: 
    def __init__(self, dim_theta, dim_y, alpha):
        self.dim_theta = dim_theta
        self.dim_y = dim_y
        self.alpha = alpha
        self.operator_fourier = self._get_operator()

    def _get_operator(self):
        i = np.linspace(1,self.dim_theta,self.dim_theta)
        ci = np.apply_along_axis(lambda x:  x**(- self.alpha), 0, i)
        return np.diag(ci)
    
    def __str__(self) -> str:
        return f'PriorCovariance(dim_theta={self.dim_theta}, dim_y={self.dim_y})'
    def __repr__(self) -> str:
         return f'PriorCovariance(dim_theta={self.dim_theta}, dim_y={self.dim_y})'
    
if __name__ == "__main__":
    prior_cov = GaussianKernel(2,2,2)
    print(prior_cov)
    print(prior_cov.operator_fourier)
#print(__name__)