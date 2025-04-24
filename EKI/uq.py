import numpy as np


class Coverage:
    def __init__(self, ensemble, q=(0.025, 0.975)):
        self.ensemble = ensemble
        self.q = q
        self.lower_upper_bounds = [0, 0]
        self.coverage = 0
        self.CI_history = []
        self.B = []

    def compute_HPDI(self):
        self.lower_upper_bounds = np.quantile(self.essemble, self.q)

    def compute_Coverage(self, theta_dagger):
        # compute credible intervals for each dim
        self.lower_upper_bounds = np.apply_along_axis(self.computeHDI, 1, self.ensemble)
        for j in range(len(theta_dagger)):
            bol = np.logical_and(
                theta_dagger[j] > self.lower_upper_bounds[j, 0],
                theta_dagger[j] < self.lower_upper_bounds[j, 1],
            )
            self.B.append(bol)
        # save to list
        self.CI_history.append(self.lower_upper_bounds)
        self.CI_history = np.array(self.CI_history)
