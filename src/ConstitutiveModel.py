import numpy as np
from scipy.optimize import root_scalar

def thermal_omega_ref(T):
    """
    Compute the reference angular frequency based on temperature T
    We use the parameters from Table 6 of Dawson et al (2027/2028)
    particularly the numbers from Kawada 1978m T > TC1
    """
    """
    Compute the reference angular frequency based on temperature T
    We use the parameters from Table 6 of Dawson et al (2027/2028)
    particularly the numbers from Kawada 1978m T > TC1
    """
    Ea = 55.3e3
    tau_0 = 5.3e-16
    omega_0 = 1 / tau_0
    kB = 1.380649e-23  # Boltzmann constant [J/K]
    omega_ref = omega_0 * np.exp(-Ea / (kB * T))
    
    return omega_ref


class DebyeModel:
    def __init__(
            self, 
            eps_infty, 
            delta_eps, 
            sigma_s, 
            omega_ref = None, 
            T = None, 
            z=None
        ):
        self.eps_0 = 8.854187817e-12  # vacuum permittivity [F/m]   

        self.eps_infty = eps_infty
        self.delta_eps = delta_eps
        self.sigma_s = sigma_s

        # T has to be a scalar
        if T is not None and not np.isscalar(T):
            raise ValueError("T must be a scalar")

        # if delta_eps is a scalar
        if np.isscalar(delta_eps):
            self.mode = 1
        else:
            self.mode = len(delta_eps)

        if T is not None:
            self.omega_ref = thermal_omega_ref(T)
        else:
            if omega_ref is None:
                raise ValueError("omega_ref must be provided if T is not supplied")
            self.omega_ref = omega_ref
        self.tau = 1 / self.omega_ref

        self.T = T
        self.z = z

        self.shape_check()

    # def shape_check(self):
    #     """
    #     if z is supplied, check that other inputs have the same shape as z
    #     """
    #     if self.z is not None:
    #         shape = self.z.shape
    #         if (self.eps_infty.shape != shape or
    #             self.delta_eps.shape != shape or
    #             self.sigma_s.shape != shape or
    #             self.eps_0.shape != shape or
    #             self.tau.shape != shape):
    #             raise ValueError("All input arrays must have the same shape as z")

    def debye_single(self, omega):
        """
        Debye model for one relaxation
        """
        eps = (self.eps_infty + 
            self.delta_eps / (1 + 1j * omega * self.tau) - 1j * self.sigma_s / (self.eps_0 * omega))
        eps_p  = (self.eps_infty + 
            self.delta_eps / (1 + (omega * self.tau)**2))
        eps_pp = (omega * self.tau * self.delta_eps / 
            (1 + (omega * self.tau)**2) + self.sigma_s / (self.eps_0 * omega))
        
        return eps, eps_p, eps_pp

    def debye_multiple(self, omega):
        raise NotImplementedError("Debye multiple relaxation is not implemented yet")


class ColeColeModel:

    def __init__(
            self, 
            eps_infty, 
            delta_eps, 
            sigma_s, 
            alpha, 
            omega_ref=None, 
            T=None, 
            z=None
        ):

        self.eps_0 = 8.854187817e-12  # vacuum permittivity [F/m]   

        self.eps_infty = eps_infty
        self.delta_eps = delta_eps
        self.sigma_s = sigma_s
        self.alpha = alpha # distribution parameter

        # T has to be a scalar
        if T is not None and not np.isscalar(T):
            raise ValueError("T must be a scalar")

        if np.isscalar(delta_eps):
            self.mode = 1
        else:
            self.mode = len(delta_eps)

        if T is not None:
            self.omega_ref = thermal_omega_ref(T)
        else:
            if omega_ref is None:
                raise ValueError("omega_ref must be provided if T is not supplied")
            self.omega_ref = omega_ref
        self.tau = 1 / self.omega_ref

        self.T = T
        self.z = z

        self.shape_check()

    # def shape_check(self):
    #     if self.z is not None:
    #         shape = self.z.shape
    #         if (self.eps_infty.shape != shape or
    #             self.delta_eps.shape != shape or
    #             self.sigma_s.shape != shape or
    #             self.alpha.shape != shape or
    #             self.eps_0.shape != shape or
    #             self.tau.shape != shape):
    #             raise ValueError("All input arrays must have the same shape as z")

    def cole_cole(self, omega):
        for m in range(self.mode):
            eps = (self.delta_eps[m] / (1 + (1j * omega * self.tau[m])**(1 - self.alpha[m])) 
                   - 1j * self.sigma_s / (self.eps_0 * omega))
        return eps


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
