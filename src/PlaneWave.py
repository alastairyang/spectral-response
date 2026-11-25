import numpy as np


class PlaneWave:
    def __init__(self, H, sigma, model, s=0):
        if H < 0:
            raise ValueError("H must be nonnegative.")
        # assuming non-magnetic
        self.eps_0 = 8.854187817e-12 # permittivity of free space
        self.mu = 4 * np.pi * 1e-7 # permeability of free space

        self.H = H
        self.s = s
        self.b = s + H # depth axis is [s, s+H], positively increasing
        self.sigma = sigma  # DC conductivity ?
        self.model = model # frequency dependent constitutive relation for real permittivity

    def compute_eps(self, omega):
        """Effective absolute permittivity, including additional conductivity."""
        omega = np.asarray(omega, dtype=float)
        if np.any(~np.isfinite(omega)) or np.any(omega <= 0):
            raise ValueError("omega must be finite and strictly positive.")

        eps_r = np.asarray(self.model(omega), dtype=complex)
        return self.eps_0 * eps_r # - 1j * self.sigma / omega

    def gamma(self, omega):
        omega = np.asarray(omega, dtype=float)
        return 1j * omega * np.sqrt(self.mu * self.compute_eps(omega))

    def alpha(self, omega):
        return self.gamma(omega).real

    def beta(self, omega):
        return self.gamma(omega).imag

    def solution(self, omega, distance):
        """One-way unit-field propagator over a distance within this layer."""
        return np.exp(-self.gamma(omega) * distance)

    def _base_solve(self, omega):
        return self.solution(omega, self.H)

    def _surface_solve(self, omega):
        return self.solution(omega, 0.0)
    def base_amplitude(self, omega):
        """
        Base return in decibels relative to the surface return.
        """
        return self.decibel(
            np.abs(self._base_solve(omega)), 
            np.abs(self._surface_solve(omega))
        )

    # def base_phase(self, omega):
    #     """
    #     Base return phase in radians relative to the surface return.
    #     """
    #     return np.angle(self._base_solve(omega) / self._surface_solve(omega))

    def base_phase(self, omega):
        """
        Directly computing unwrapped phase in radians relative to the surface return.
        account for one way travel
        """
        return -self.gamma(omega).imag * self.H
    @staticmethod
    def decibel(val1, val2):
        val1 = np.asarray(val1, dtype=float)
        val2 = np.asarray(val2, dtype=float)
        if np.any(val1 <= 0) or np.any(val2 <= 0):
            raise ValueError("Values must be strictly positive.")
        return 20 * np.log10(val1 / val2)