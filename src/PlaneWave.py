import numpy as np


class PlaneWave:
    def __init__(self, H, mu, sigma, eps_0,model):
        if H < 0:
            raise ValueError("H must be nonnegative.")

        self.H = H
        self.mu = mu
        self.sigma = sigma  # Additional conductivity only
        self.model = model
        self.eps_0 = eps_0

    def gamma(self, omega):
        omega = np.asarray(omega, dtype=float)

        if np.any(~np.isfinite(omega)) or np.any(omega <= 0):
            raise ValueError("omega must be finite and strictly positive.")

        # constitutive relation for the material
        eps_r = np.asarray(self.model(omega), dtype=complex)
        eps_abs = self.eps_0 * eps_r

        argument = (
            -(omega**2) * self.mu * eps_abs
            + 1j * omega * self.mu * self.sigma
        )

        # For a passive material with this convention, the principal
        # square root gives nonnegative attenuation.
        return np.sqrt(argument.astype(complex))

    def alpha(self, omega):
        return self.gamma(omega).real

    def beta(self, omega):
        return self.gamma(omega).imag

    def bed_transfer(self, omega):
        return np.exp(-self.gamma(omega) * self.H)
