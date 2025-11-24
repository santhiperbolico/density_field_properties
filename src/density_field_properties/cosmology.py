import numpy as np

RHO_C0 = 2.775e11


class Cosmology:
    def __init__(self, omega_matter: float, omega_lambda: float, h0: float):
        self.omega_matter = omega_matter
        self.h0 = h0
        self.omega_lambda = omega_lambda
        self.rho_c_h2_msun_mpc3 = RHO_C0 * (h0**2)

    def convert_m200b_to_r200b(
        self,
        m200b: np.ndarray,
        z: float = 0.0,
    ) -> np.ndarray:
        """
        Converts the mass, defined using 200 times the critical density of the universe
        (m200b), to r200b, the radius within which the mass is enclosed. This calculation
        assumes a spherical overdensity using cosmological parameters.

        Parameters
        ----------
        m200b : np.ndarray
            Mass defined using 200 times the critical density of the universe, in units
            Msun h^-1. Expected as an array for input flexibility.
        z: float, optional
            Redshift of the simulation. Default is 0.

        Returns
        -------
        np.ndarray
            Corresponding r200b values calculated from the input m200b array, expressed
            in Mpc/h
        """
        rho_mean = self.omega_matter * self.rho_c_h2_msun_mpc3 * (1 + z) ** 3
        r200b = (3 * m200b / (4 * np.pi * 200 * rho_mean)) ** (1 / 3)
        return r200b

    def convert_r200b_to_m200b(self, r200b: np.ndarray) -> np.ndarray:
        """
        Convert spherical overdensity mass from r200b to m200b.

        This function calculates the m200b mass, which represents the spherical
        overdensity mass defined as 200 times the critical density of the
        universe, based on the input r200b (radius corresponding to 200 times
        the critical density). The calculation incorporates the critical density
        and matter density values for the universe.

        Parameters
        ----------
        r200b : np.ndarray
            The radius corresponding to the spherical overdensity mass
            defined as 200 times the critical density, in units of Mpc/h.

        Returns
        -------
        np.ndarray
            The spherical overdensity mass m200b in units of solar masses (Msun/hh).
        """
        return r200b**3 * (4.0 / 3.0 * np.pi * self.rho_c_h2_msun_mpc3 * self.omega_matter)
