import numpy as np


class PlaneWave:
    def __init__(self, H, mu, sigma, model, s=0):
        if H < 0:
            raise ValueError("H must be nonnegative.")

        self.H = H
        self.s = s
        self.b = s + H # depth axis is [s, s+H], positively increasing
        self.mu = mu
        self.sigma = sigma  # Additional conductivity only
        self.model = model # frequency dependent constitutive relation for real permittivity
        self.eps_0 = 8.854187817e-12 # permittivity of free space

    def compute_eps(self, omega):
        """
        Compute the absolute electrical permittivity
        """
        omega = np.asarray(omega, dtype=float)
        if np.any(~np.isfinite(omega)) or np.any(omega <= 0):
            raise ValueError("omega must be finite and strictly positive.")

        eps_r = np.asarray(self.model(omega), dtype=complex)
        return self.eps_0 * eps_r

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

    def solution(self, omega, z):
        gamma = self.gamma(omega)
        return np.exp(-gamma * z)
    def bed_amplitude(self, omega):
        """
        Bed return in decibels relative to the surface return.
        """
        return self.decibel(
            np.abs(self._bed_solve(omega)), 
            np.abs(self._surface_solve(omega))
        )

    def bed_phase(self, omega):
        """
        Bed return phase in radians relative to the surface return.
        """
        return np.angle(self._bed_solve(omega) / self._surface_solve(omega))

    def _bed_solve(self, omega):
        return self.solution(omega, self.b)

    def _surface_solve(self, omega):
        return self.solution(omega, self.s)

    @staticmethod
    def decibel(val1, val2):
        val1 = np.asarray(val1, dtype=float)
        val2 = np.asarray(val2, dtype=float)
        if np.any(val1 <= 0) or np.any(val2 <= 0):
            raise ValueError("Values must be strictly positive.")
        return 20 * np.log10(val1 / val2)