import numpy as np
import matplotlib.pyplot as plt

class LayeredColumn:
    """
    Layered column model for electromagnetic wave propagation.
    We consider wave propagation at normal incidence
    """
    def __init__(self, waves):
        # waves have to be list of PlaneWave instances
        if not isinstance(waves, list):
            raise ValueError("waves must be a list of PlaneWave instances.")
        for wave in waves:
            if not hasattr(wave, "gamma"):
                raise ValueError("Each item in waves must be a PlaneWave instance.")

        self.waves = waves
        self.c = 299_792_458.0
        self._check_geometry_consistency()

        self.surface = self.waves[0].s
        self.base = self.waves[-1].b
        print(f"Surface: {self.surface}, Base: {self.base} m below surface")
        self.n_interface = len(self.waves) - 1

    def _check_geometry_consistency(self):
        """
        check if the wave.b of layer i matches the wave.s of the next instance
        """
        for i in range(len(self.waves) - 1):
            if not np.allclose(self.waves[i].b, self.waves[i+1].s):
                raise ValueError(f"Geometry inconsistency between layer {i} and layer {i+1}.")

    def _shift_geometry(self):
        """
        shift the geometry of the individual plane wave object
        so that their surface are all zero
        """
        for wave in self.waves:
            shift = wave.s
            wave.s -= shift
            wave.b -= shift
        return
    def show_geometry(self):
        """
        Display the geometry of the layered column.
        """
        for i, wave in enumerate(self.waves):
            print(f"Layer {i+1}: s = {wave.s}, b = {wave.b}")

    def show_permittivity(self, omega):
        """
        Show the relative electric permittivity of the structure
        Plot the relative electric permittivity of each layer 
        as a function of frequency.
        
        """
        freq_colors = plt.cm.viridis(np.linspace(0, 1, len(omega)))
        for i, wave in enumerate(self.waves):
            for j, w in enumerate(omega):
                # plot permittivity (x) vs depth (y)
                # and each frequency has different color
                eps_vector = np.ones(2) * np.abs(wave.compute_eps(w)/wave.eps_0)
                plt.plot(eps_vector, 
                         [wave.s, wave.b], 
                         color=freq_colors[j]
                         )
        # draw horizontal lines for layer boundaries
        for wave in self.waves:
            plt.axhline(y=wave.s, color='k', linestyle='--', linewidth=0.5)
            plt.axhline(y=wave.b, color='k', linestyle='--', linewidth=0.5)
            # add layer number in the middle of the layer
            mid = (wave.s + wave.b) / 2
            # plt.text(0, mid, f"Layer {self.waves.index(wave)+1}", va='center', ha='left', fontsize=8)
        # add one legend showing the color for frequencies
        for j, w in enumerate(omega):
            f = w / (2 * np.pi)
            plt.plot([], [], color=freq_colors[j], label=f"{f/1e6:.1f} MHz")
        plt.gca().invert_yaxis()  # invert y-axis to have depth increasing downwards
        plt.xlabel("Relative Permittivity")
        plt.ylabel("Depth")
        plt.legend()
        plt.show()
        return

    def reflection_coefficient(self, omega):
        """
        Compute the reflection coefficient at normal incidence 
        for each interface
        """
        r = np.zeros((self.n_interface, len(omega)), dtype=complex)
        for i in range(self.n_interface):
            eps_i = self.waves[i].compute_eps(omega)
            eps_ip1 = self.waves[i+1].compute_eps(omega)
            for j, (eps_i_val, eps_ip1_val) in enumerate(zip(eps_i, eps_ip1)):
                r[i, j] = (np.sqrt(eps_i_val) - np.sqrt(eps_ip1_val)) / (np.sqrt(eps_ip1_val) + np.sqrt(eps_i_val))
        return r

    def transmission_coefficient(self, omega):
        """
        Compute the transmission coefficient at normal incidence 
        for each interface
        """
        T = np.zeros((self.n_interface, len(omega)), dtype=complex)
        for i in range(self.n_interface):
            eps_i = self.waves[i].compute_eps(omega)
            eps_ip1 = self.waves[i+1].compute_eps(omega)
            for j, (eps_i_val, eps_ip1_val) in enumerate(zip(eps_i, eps_ip1)):
                T[i, j] = 2 * np.sqrt(eps_i_val) / (np.sqrt(eps_i_val) + np.sqrt(eps_ip1_val))
        return T

    def attenuation(self, omega):
        """
        Compute the amplitude attenuation through each layer
        """
        omega = np.asarray(omega, dtype=float)
        att = np.zeros((len(self.waves), len(omega)), dtype=float)
        for i, wave in enumerate(self.waves):
            att[i] = np.exp(-wave.gamma(omega).real * (wave.b - wave.s))
        return att

    def attenuation_rate(self, omega):
        """
        Full column depth averaged attenuation rate.
        
        We compute the depth-averaged attenuation rate
        via the attenuation constant

        """
        omega = np.asarray(omega, dtype=float)
        ice_layers = self.waves[1:-1]
        att_ice = self.attenuation(omega)[:-1]
        total_ice_thickness = sum(w.b - w.s for w in ice_layers)

        amp_at_bed = np.prod(att_ice, axis=0)
        loss_dB = -20.0 * np.log10(amp_at_bed)  # positive dB loss
        Na = loss_dB * 1e3 / total_ice_thickness  # dB/km
        return Na



    def returns(self, omega, incident_field=1.0):
        """
        Return (reflected_field, downward_transmitted_field) for each interface.

        Reflected fields are referenced to the top of the column.
        Transmitted fields are evaluated immediately below each interface.
        Internal multiples are not included.

        Returns
        -------
        echoes : list of tuples
            Each tuple contains (reflected_field, downward_transmitted_field) for an interface.
            The reflected field is referenced to the top of the column, and the transmitted
            field is evaluated immediately below the interface.
        """
        omega = np.asarray(omega, dtype=float)
        r = self.reflection_coefficient(omega)
        t_down = self.transmission_coefficient(omega)
        t_up = 1.0 - r  # Reverse electric-field transmission: 1 + r_reverse

        down = np.broadcast_to(
            np.asarray(incident_field, dtype=complex), omega.shape
        ).copy()
        up = np.ones_like(down)
        echoes = []

        for i, wave in enumerate(self.waves[:-1]):
            P = np.exp(-wave.gamma(omega) * (wave.b - wave.s))

            down_at_interface = down * P
            reflected = down_at_interface * r[i] * P * up
            transmitted = down_at_interface * t_down[i]
            echoes.append((reflected, transmitted))

            down = transmitted
            up = t_up[i] * P * up

        return echoes

    def spectral_ratio(self, omega, n_ref, shallow=0, deep=1):
        """
        Compute the deep/shallow first-arrival spectral ratio.

        omega: strictly increasing angular frequencies, in rad/s.
        n_ref: scalar reference refractive index, or one index per layer
               between the two reflecting interfaces, in shallow-to-deep order.
        shallow, deep: zero-based interface indices into returns().
        """
        import warnings
        omega = np.asarray(omega, dtype=float)

        if not 0 <= shallow < deep < self.n_interface:
            raise ValueError("Require 0 <= shallow < deep < n_interface.")

        # Interface i is at the bottom of waves[i]. Only the layers
        # between the two reflecting interfaces add differential delay.
        layers = self.waves[shallow + 1 : deep + 1]
        thickness = np.array([wave.H for wave in layers], dtype=float)
        n_ref = np.broadcast_to(np.asarray(n_ref, dtype=float), thickness.shape)
        if np.any(~np.isfinite(n_ref)) or np.any(n_ref <= 0):
            raise ValueError("Reference indices must be finite and positive.")

        tau_ref_s = 2.0 * np.dot(thickness, n_ref) / self.c

        echoes = self.returns(omega)
        E_shallow = echoes[shallow][0]
        E_deep = echoes[deep][0]

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = E_deep / E_shallow

        invalid = ~np.isfinite(ratio) | (ratio == 0)
        if np.any(invalid):
            warnings.warn(
                f"Ratio contains {np.sum(invalid)} zero or nonfinite values; "
                "masking them as NaN.",
                RuntimeWarning,
            )
            ratio = np.where(invalid, np.nan + 1j * np.nan, ratio)

        # Our forward convention is exp(-i * omega * tau).
        # Remove the reference ramp before unwrapping.
        corrected = ratio * np.exp(1j * omega * tau_ref_s)

        # Unwrap phase only across valid (finite) points
        phase_corrected_rad = np.full_like(omega, np.nan, dtype=float)
        valid = np.isfinite(corrected)
        if np.any(valid):
            phase_corrected_rad[valid] = np.unwrap(np.angle(corrected[valid]))

        with np.errstate(divide="ignore", invalid="ignore"):
            amplitude_db = 20.0 * np.log10(np.abs(ratio))

        return {
            "ratio": ratio,
            "amplitude_db": amplitude_db,
            "phase_corrected_rad": phase_corrected_rad,
            "tau_ref_s": tau_ref_s,
        }