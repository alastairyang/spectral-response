import numpy as np
from scipy.special import erf


def robin_temperature(
    z, H, T_surface, accumulation_m_per_yr, q_b,
    *, k=2.1, rho=917.0, cp=2009.0,
):
    """Steady Robin ice temperature at depth z (m), positive downward.

    H: ice thickness (m)
    T_surface: surface temperature (°C or K)
    accumulation_m_per_yr: nonnegative ice-equivalent accumulation (m/yr)
    q_b: upward basal heat flux into the ice (W/m²)
    k, rho, cp: constant thermal conductivity, density, and specific heat (SI)
    """
    z = np.asarray(z, dtype=float)

    if H <= 0 or k <= 0 or rho <= 0 or cp <= 0:
        raise ValueError("H, k, rho, and cp must be positive.")
    if accumulation_m_per_yr < 0:
        raise ValueError("Accumulation must be nonnegative.")
    if np.any(~np.isfinite(z)) or np.any((z < 0) | (z > H)):
        raise ValueError("Depths must be finite and lie in [0, H].")

    a = accumulation_m_per_yr / (365.25 * 24 * 3600)
    kappa = k / (rho * cp)

    if a == 0:
        return T_surface + (q_b / k) * z

    lam = np.sqrt(a / (2 * kappa * H))
    return T_surface + (q_b / k) * (np.sqrt(np.pi) / (2 * lam)) * (
        erf(lam * H) - erf(lam * (H - z))
    )
