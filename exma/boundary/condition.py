import os
import sysconfig
import ctypes as ct
import numpy as np

suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None: suffix = ".so"

boundary_dir = os.path.dirname(__file__)
boundary_name = "lib_boundary" + suffix
libboundary = os.path.abspath(os.path.join(boundary_dir, boundary_name))
lib_boundary = ct.CDLL(libboundary)


class condition:
    """
    apply some boundary condition to the particles

    Parameters
    ----------
    box_size : numpy array of three floats
        box size in x, y, z
    """
    
    def __init__(self, box_size):
        
        self.box_size = box_size


class periodic(condition):
    """
    apply periodic boundary condition to the particles
    """

    def pbc(self, natoms, positions):
        """
        applies periodic boundary conditions to the particles and returns the
        positions in [0, box_size)
        
        Parameters
        ----------
        natoms : integer
            number of atoms

        positions : numpy array with float32 data
            unwrapped positions in the SoA convention

        Returns
        -------
        positions : numpy array with float32 dat
            wrapped positions
        """

        box_size = self.box_size.astype(np.float32)
        box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))

        positions = positions.astype(np.float32)
        x_C = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))
        
        pbc_c = lib_boundary.pbc
        pbc_c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p]

        pbc_c(natoms, box_size, x_C)
        # pbc_c operates over the object x_C that has the positions, they are
        # modificated in a void function so there is not a returned value
        # or some to read from buffer because the information needed is in
        # positions
        
        return positions


class minimum_image(condition):
    """
    minimum image
    """

    def minimum_image(self, x_central, x_interact):
        """
        nearest particle in minimum image convention

        Parameters
        ----------
        x_central : numpy array with three floats
            coordinates of the central atom

        x_interact : numpy array with three floats
            coordinates of the interact atom

        Returns
        -------
        x_ci : numpy array with three float32
            components of the vector that separates the atom interact from the
            atom central
        """

        box_size = self.box_size.astype(np.float32)
        box_C = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))
        
        x_central = np.array(x_central, dtype=np.float32)
        central_C = x_central.ctypes.data_as(ct.POINTER(ct.c_void_p))
        
        x_interact = np.array(x_interact, dtype=np.float32)
        interact_C = x_interact.ctypes.data_as(ct.POINTER(ct.c_void_p))

        x_ci = np.zeros(3, dtype=np.float32)
        ci_C = x_ci.ctypes.data_as(ct.POINTER(ct.c_void_p))
        
        minimum_image_c = lib_boundary.minimum_image
        minimum_image_c.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                    ct.c_void_p]

        minimum_image_c(box_C, central_C, interact_C, ci_C)
        # minimum_image_c operates over the object ci_C that has the distance, 
        # that is modificated in a void function so there is not a returned value
        # or some to read from buffer because the information needed is in x_ci

        return x_ci
