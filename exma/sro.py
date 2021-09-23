import numpy as np
from scipy import integrate


class sro:
    """
    class with short range order parameter
    """


class amorphous(sro):
    """
    amourphous parameter to characterize the short range ordering, defined in
    https://doi.org/10.1039/D1CP02216D, using the itegration of the radial
    distribution function for diatomic systems

    Parameters
    ----------
    rdf_x : numpy array
        x of the radial distribution function

    rdf_y : numpy array
        y of the radial distribution function
    """

    def __init__(self, rdf_x, rdf_y):
        self.rdf_x = rdf_x
        self.rdf_y = rdf_y

    def parameter(self, rcut):
        """
        Parameters
        ----------
        rcut : float
            cutoff radius

        Returns
        -------
        sro : float
            short range order, amorphous parameter
        """
        mask = self.rdf_x < rcut
        vol = (4.0 / 3.0) * np.pi * np.power(rcut, 3)

        ix = self.rdf_x[mask]
        iy = 4.0 * np.pi * ix * ix * self.rdf_y[mask]

        cab = integrate.simps(iy, ix)
        sro = np.log(cab / vol)

        return sro
