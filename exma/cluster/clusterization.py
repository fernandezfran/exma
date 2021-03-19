import os
import sysconfig
import ctypes as ct
import numpy as np
from sklearn.cluster import DBSCAN

suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None: suffix = ".so"

cluster_dir = os.path.dirname(__file__)
cluster_name = "lib_cluster" + suffix
libcluster = os.path.abspath(os.path.join(cluster_dir, cluster_name))
lib_cluster = ct.CDLL(libcluster)


class dbscan:
    """
    the main objetive of this module is to accomodate data (calculate the
    distance matrix taking account of the PBC) before using sklearn.cluster.DBSCAN
    (density-based spatial clustering of applications with noise)

    see: https://scikit-learn.org/stable/modules/clustering.html
    and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

    Parameters
    ----------
    eps : float
        like an rcut where an atoms stop to be considered part of a cluster

    min_samples : int, default=2
        the number of atoms that can be a core point
        by default, 2 atoms can be a cluster
    """

    def __init__(self, eps, min_samples=2):

        self.eps = eps
        self.min_samples = min_samples

        self.distance_matrix_c = lib_cluster.distance_matrix
        self.distance_matrix_c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                                           ct.c_void_p]


    def of_this_frame(self, box_size, atom_type, positions, atom_type_c):
        """
        obtain the dbscan clusterization of the actual frame

        Parameters
        ----------
        box_size : numpy array with three floats
            the box size in x, y, z
        
        atom_type : numpy array with integers (could be char)
            type of atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        
        atom_type_c : integer (or char)
            type of atom to which you want to perform the cluster analysis

        Returns
        -------
        positions_c : numpy array with float32 data
            the positions in the SoA convention of the atoms that are
            considered in the cluster analysis

        id_cluster : array of ints
            the id number of the cluster to which belongs the corresponding
            atom (the array is ordered)
            it is equal to -1 if the atom is isolated
        """

        xyz = np.split(positions, 3)
        x, y, z = xyz[0], xyz[1], xyz[2]
        x, y, z = x[atom_type == atom_type_c], y[atom_type == atom_type_c], \
                  z[atom_type == atom_type_c]

        positions_c = np.concatenate((x, y, z)).astype(np.float32)
        natoms_c = np.intc(len(x))
        distrix = np.zeros(natoms_c * natoms_c, dtype=np.float32)

        # prepare data to C function
        box_size = box_size.astype(np.float32)
        box_C = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))

        x_C = positions_c.ctypes.data_as(ct.POINTER(ct.c_void_p))

        distrix_C = distrix.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.distance_matrix_c(natoms_c, box_C, x_C, distrix_C)
        # a void function that modifies the values of distrix

        distrix = distrix.reshape((natoms_c, natoms_c))
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, \
                metric='precomputed').fit(distrix)
        
        id_cluster = np.asarray(db.labels_)

        return positions_c, id_cluster
