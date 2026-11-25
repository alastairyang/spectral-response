import numpy as np
from scipy.optimize import root_scalar
import torch

def thermal_omega_ref(T, tau_0):
    if not np.isscalar(T):
        raise ValueError("T must be a scalar")

    Ea = 55.3e3
    R = 8.314  # gas constant [J/(mol*K)]
    tau_0 = np.asarray(tau_0, dtype=float)
    tau = tau_0 * np.exp(Ea / (R * T))
    return tau


class ColeColeModel:
    def __init__(
            self, 
            eps_infty, 
            delta_eps, 
            sigma_s, 
            alpha, 
            tau_0 = 5.3e-16, # Table 6 in Eliza's review paper
            T=None, 
            z=None
        ):
        self.eps_0 = 8.854187817e-12  # vacuum permittivity [F/m]
        self.eps_infty = eps_infty
        self.delta_eps = np.atleast_1d(delta_eps)
        self.sigma_s = sigma_s
        self.alpha = np.atleast_1d(alpha)

        if T is not None and not np.isscalar(T):
            raise ValueError("T must be a scalar")

        self.mode = len(self.delta_eps)

        if T is not None:
            self.tau_0 = thermal_omega_ref(T, tau_0) * np.ones(self.mode)
        else:
            self.tau_0 = np.atleast_1d(tau_0)

        self.T = T
        self.z = z

    def cole_cole(self, omega, T=None):
        """
        Eq. 38 (Sect 5.1) in Dawson et al (2027/2028)

        """
        if T is not None:
            tau_0 = 5.3e-16
            self.tau_0 = thermal_omega_ref(T, tau_0) * np.ones(self.mode)
            self.T = T

        eps = 0 + 0j
        for m in range(self.mode):
            eps += (self.delta_eps[m] /
                    (1 + (1j * omega * self.tau_0[m])**(1 - self.alpha[m])))

        eps += self.eps_infty
        eps -= 1j * self.sigma_s / (self.eps_0 * omega)
        return eps

    def grad_imag(self, omega):
        """
        Gradient of the imaginary part wrt angular frequency.
        Uses PyTorch autograd (note: in this class eps = eps' - j*eps'',
        so eps.imag is negative).
        """
        is_scalar = np.isscalar(omega)
        omega_t = torch.as_tensor(omega, dtype=torch.float64).clone().detach().requires_grad_(True)

        eps = self.cole_cole(omega_t)
        (-eps.imag).sum().backward()

        grad = omega_t.grad.detach().numpy()
        return grad.item() if is_scalar else grad


    def loss_tangent(self, omega):
        eps = self.cole_cole(omega)
        return -eps.imag / eps.real 

    
class DebyeModel(ColeColeModel):

    """
    Debye model as a special case of Cole Cole model
    """
    def __init__(
            self, 
            eps_infty, 
            delta_eps, 
            sigma_s, 
            tau_0 = 5.3e-16, 
            T=None, 
            z=None
        ):
        # Debye is Cole-Cole with alpha = 0 for all modes
        alpha = np.zeros_like(np.atleast_1d(delta_eps), dtype=float)
        super().__init__(
            eps_infty=eps_infty,
            delta_eps=delta_eps,
            sigma_s=sigma_s,
            alpha=alpha,
            tau_0=tau_0,
            T=T,
            z=z
        )

    def debye(self, omega):
        """
        Returns (eps, eps_p, eps_pp) for single or multiple relaxations.
        """
        eps = self.cole_cole(omega)
        return eps

    def loss_tangent(self, omega):
        eps = self.debye(omega)
        return -eps.imag / eps.real 

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
