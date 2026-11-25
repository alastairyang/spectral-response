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

    Note that since our Cole Cole model (or any contitutive model)
    already includes the static conductivity term, while in the paper
    we didn't, this means that the imaginary part of the permittivity
    already includes the contribution from the static conductivity.

    Our implementation takes this into account when computing the attenuation factor.
    """
    def __init__(self, omega, model, T=None):
        self.model = model # constitutive model (e.g. Debye)

        self.eps_infty = self.model.eps_infty
        self.sigma_s   = self.model.sigma_s
        self.eps_0     = self.model.eps_0
        self.c         = 3e8 

        self.omega = np.asarray(omega, dtype=float)
        self.torch_omega = torch.tensor(self.omega, dtype=torch.float64)
        self.freq  = self.omega / (2 * np.pi)
        if T is None:
            self.T = 240
            print("No temperature provided; set to default T = 240 K")
        else:
            self.T = T
            

    def alpha_bias(self):
        """
        Compute the bias in estimating alpha (attenuation factor)
        using spectral ratio method

        """
        depspp_dw = self.model.grad_imag(self.omega)

        prefactor = 1 / (2 * self.c * self.eps_0 * np.sqrt(self.eps_infty))
        return prefactor * (self.omega**2 * depspp_dw * self.eps_0)

    def alpha_SR_estimate(self, T=None):
        """
        Theoretical estimate of alpha (attenuation factor) using spectral ratio method
        """
        depspp_dw = self.model.grad_imag(self.omega)

        if T is not None:
            eps = self.model.cole_cole(self.omega, T=T)
        else:
            eps = self.model.cole_cole(self.omega, T=self.T)

        eps_pp = -eps.imag

        prefactor = 1 / (2 * self.c * self.eps_0 * np.sqrt(self.eps_infty))
        return prefactor * (self.omega * eps_pp * self.eps_0 + self.omega**2 * depspp_dw * self.eps_0)

    def alpha(self, T=None):
        """
        Theoretical estimate of alpha (attenuation factor) from plane wave
        """
        if T is not None:
            eps = self.model.cole_cole(self.omega, T=T)
        else:
            eps = self.model.cole_cole(self.omega, T=self.T)

        eps_pp = -eps.imag

        prefactor = 1 / (2 * self.c * self.eps_0 * np.sqrt(self.eps_infty))
        return prefactor * (self.omega * eps_pp * self.eps_0) # sigma_s was already in Cole-Cole model

    def _cole_cole_torch(self, omega_t, T_t):
        """PyTorch-differentiable Cole-Cole permittivity wrt both omega and T."""
        Ea, R, tau_0_base = 55.3e3, 8.314, 5.3e-16
        tau = tau_0_base * torch.exp(Ea / (R * T_t))

        eps = torch.zeros_like(omega_t, dtype=torch.complex128)
        for m in range(self.model.mode):
            delta_eps = float(self.model.delta_eps[m])
            alpha_m = float(self.model.alpha[m])
            eps = eps + delta_eps / (1.0 + (1j * omega_t * tau) ** (1.0 - alpha_m))

        eps = eps + float(self.eps_infty)
        eps = eps - 1j * float(self.sigma_s) / (self.eps_0 * omega_t)
        return eps

    def grad_T(self, T=None):
        """
        Compute d(alpha_SR)/dT and d(alpha_bias)/dT across self.omega using PyTorch AD.
        Returns (d_alpha_SR_dT, d_alpha_bias_dT) as NumPy arrays [Np/(m*K)].
        """
        T_val = self.T if T is None else T
        T_t = torch.tensor(float(T_val), dtype=torch.float64, requires_grad=True)
        omega_t = self.torch_omega.clone().detach().requires_grad_(True)

        eps = self._cole_cole_torch(omega_t, T_t)
        eps_pp = -eps.imag

        # First derivative wrt omega (keep graph for second derivative wrt T)
        depspp_dw = torch.autograd.grad(eps_pp.sum(), omega_t, create_graph=True)[0]

        prefactor = 1.0 / (2.0 * self.c * self.eps_0 * np.sqrt(self.eps_infty))
        alpha_sr = prefactor * (omega_t * eps_pp * self.eps_0 + omega_t**2 * depspp_dw * self.eps_0)
        alpha_bias = prefactor * (omega_t**2 * depspp_dw * self.eps_0 - float(self.sigma_s))

        # Differentiate each frequency point wrt scalar T_t
        d_sr_dT = np.array([
            torch.autograd.grad(val, T_t, retain_graph=True)[0].item()
            for val in alpha_sr
        ])
        d_bias_dT = np.array([
            torch.autograd.grad(val, T_t, retain_graph=True)[0].item()
            for val in alpha_bias
        ])

        return d_sr_dT, d_bias_dT
