import numpy as np
import torch

class SpectralRatioApprox:
    """
    Attenuation constant inversion via spectral method
    From Dawson et al., 2026, ice sheet attenuation from radar sounding
    in the frequency domain
    """
    def __init__(self, omega, dt):
        self.omega = np.asarray(omega, dtype=float)
        self.dt = float(dt) # time interval between the surface and bed
        self.v = 3e8 / np.sqrt(3.15) # velocity in ice

        self.mu = 4 * np.pi * 1e-7 # permeability of free space
        self.eps_0 = 8.854187817e-12 # permittivity of free space
        self.eps = 3.15 * self.eps_0 # permittivity of ice

    def Q(self, ratio):
        """
        Compute the Quality factor Q from the spectral ratio.
        """

        # linear fit of ln(|ratio|) vs omega
        self._shape_check(ratio)


        ln_abs_ratio = np.log(np.abs(ratio))
        A = np.vstack([self.omega, np.ones_like(self.omega)]).T
        slope, _ = np.linalg.lstsq(A, ln_abs_ratio, rcond=None)[0]
        Q = - self.dt / (2 * slope)
        return Q

    def alpha_original(self, ratio):
        """
        Compute the attenuation constant from Q factor
        Full expression
        """
        self._shape_check(ratio)

        Q = self.Q(ratio)
        loss_tangent = 1 / Q

        inside = np.sqrt(1 + loss_tangent**2)
        alpha = self.omega * np.sqrt((1/2)* self.mu * self.eps * (inside - 1))
        return alpha

    def alpha_low_loss(self, ratio):
        """
        Compute the attenuation constant from Q factor
        Low-loss approximation
        """
        self._shape_check(ratio)

        Q = self.Q(ratio)
        return self.omega / (2 * self.v * Q)

    def attenuation_rate(self, ratio):
        """
        Compute the attenuation rate (alpha * v)
        """
        Na = self.alpha_original(ratio) * (8686) # dB/km, eq 8 in Dawson et al., 2026
        Na_low_loss = self.alpha_low_loss(ratio) * (8686) # dB/km, eq 8 in Dawson et al., 2026
        return Na, Na_low_loss

    def _shape_check(self, ratio):
        """
        Check if the shape of the input ratio matches the shape of omega.
        """
        if ratio.shape != self.omega.shape:
            raise ValueError("Shape of ratio does not match shape of omega.")


class SpectralRatioDiagnostic:
    """
    Class for quantifying the bias and identifiability of of 
    spectral ratio method.
    """
    def __init__(self, omega, model):
        self.model = model # constitutive model (e.g. Debye)

        self.eps_infty = self.model.eps_infty
        self.sigma_s   = self.model.sigma_s
        self.eps_0     = self.model.eps_0
        self.c         = 3e8 

        self.omega = np.asarray(omega, dtype=float)
        self.torch_omega = torch.tensor(self.omega, dtype=torch.float64)
        self.freq  = self.omega / (2 * np.pi)

    def alpha_bias(self):
        """
        Compute the bias in estimating alpha (attenuation factor)
        using spectral ratio method
        """
        depspp_dw = self.model.grad_imag(self.omega)

        prefactor = 1 / (2 * self.c * self.eps_0 * np.sqrt(self.eps_infty))
        return prefactor * (self.omega**2 * depspp_dw * self.eps_0 - self.sigma_s)


