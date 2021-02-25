import numpy as np
from . import boundary
from sklearn.cluster import DBSCAN

class clusterization:
    """
    cluster identification using scikit-learn lib
    """

class cluster(clusterization):
    """
    the main objetivo of this module is to accomodate data (calculate the
    distance matrix taking account of the PBC) before using sklearn.clustering
    """

    def __init__(self, box_size, eps, min_samples=2):
        """
        Parameters
        ----------

        box_size : numpy array with three floats
            the box size in x, y, z

        eps : float
            like an rcut where an atoms stop to be considered part of a cluster

        min_samples : int, default=2
            the number of atoms that can be a core point
            by default, 2 atoms can be a cluster
        """
        self.box_size = box_size
        self.eps = eps
        self.min_samples = min_samples
        
        self.bound = boundary.apply(self.box_size)


    def dbscan(self, atom_type, positions, atom_type_c):
        """
        DBSCAN (density-based spatial clusterin of applications with noise)

        see: https://scikit-learn.org/stable/modules/clustering.html
        and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

        Parameters
        ----------
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

        positions_c = np.concatenate((x, y, z))
        natoms_c = len(x)
        distrix = np.zeros(natoms_c * natoms_c)

        for i in range(0, natoms_c - 1):

            ri = [positions[i + k*natoms_c] for k in range(0,3)]

            for j in range(i + 1, natoms_c):
           
                rj = [positions[j + k*natoms_c] for k in range(0,3)]

                rij = self.bound.minimum_image(ri, rj)
                r2 = np.linalg.norm(rij)

                # the distance matrix is symetric
                distrix[natoms_c * i + j] = r2
                distrix[natoms_c * j + i] = r2

        distrix = distrix.reshape((natoms_c, natoms_c))
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, \
                metric='precomputed').fit(distrix)
        
        id_cluster = np.asarray(db.labels_)

        return positions_c, id_cluster
