import numpy as np
from scipy.optimize import root_scalar

class DebyeModel:
    def __init__(self, eps_infty, delta_eps, sigma_s, eps_0, tau, z=None):
        self.eps_infty = eps_infty
        self.delta_eps = delta_eps
        self.sigma_s = sigma_s
        self.eps_0 = eps_0
        self.tau = tau
        self.z = z

        self.shape_check()

    def shape_check(self):
        """
        if z is supplied, check that other inputs have the same shape as z
        """
        if self.z is not None:
            shape = self.z.shape
            if (self.eps_infty.shape != shape or
                self.delta_eps.shape != shape or
                self.sigma_s.shape != shape or
                self.eps_0.shape != shape or
                self.tau.shape != shape):
                raise ValueError("All input arrays must have the same shape as z")

    def debye_single(self, omega):
        """
        Debye model for one relaxation
        """
        eps = self.eps_infty + self.delta_eps / (1 + 1j * omega * self.tau) - 1j * self.sigma_s / (self.eps_0 * omega)
        eps_p  = self.eps_infty + self.delta_eps / (1 + (omega * self.tau)**2)
        eps_pp = omega * self.tau * self.delta_eps / (1 + (omega * self.tau)**2) + self.sigma_s / (self.eps_0 * omega)
        
        return eps, eps_p, eps_pp

class MixingModel:

    def __init__(self, inclusion_model, host_model, fraction):
        self.inclusion_model = inclusion_model
        self.host_model = host_model
        self.fraction = fraction

    def maxwell_garnett(self, omega):
        eps_i = self.inclusion_model(omega)
        eps_h = self.host_model(omega)
        phi = self.fraction

        return eps_h * (
            eps_i + 2 * eps_h + 2 * phi * (eps_i - eps_h)
        ) / (
            eps_i + 2 * eps_h - phi * (eps_i - eps_h)
        )
