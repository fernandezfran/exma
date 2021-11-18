import ctypes as ct
import os
import sysconfig

import numpy as np

suffix = sysconfig.get_config_var("EXT_SUFFIX")
if suffix is None:
    suffix = ".so"

cn_dir = os.path.dirname(__file__)
cn_name = "lib_cn" + suffix
libcn = os.path.abspath(os.path.join(cn_dir, cn_name))
lib_cn = ct.CDLL(libcn)


class monoatomic:
    """
    cn of a monoatomic system

    Parameters
    ----------
    natoms : integer
        number of atoms

    rcut_e : float
        external of the shell

    rcut_i : float
        internal of the shell
    """

    def __init__(self, natoms, rcut_e, rcut_i=0.0):

        self.natoms = natoms
        self.rcut_e = rcut_e
        self.rcut_i = rcut_i

        self.cn_ = np.zeros(self.natoms, dtype=np.intc)
        self.ncn_ = 0

        self.cn_c = lib_cn.monoatomic
        self.cn_c.argtypes = [
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_float,
            ct.c_float,
            ct.c_void_p,
        ]
        self.cn_res = (ct.c_int * natoms)()

    def accumulate(self, box_size, positions):
        """
        Parameters
        ----------
        box_size : numpy array with three floats
            the box size in x, y, z

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        box_size = box_size.astype(np.float32)
        box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))

        positions = positions.astype(np.float32)
        x_c = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.cn_c(
            self.natoms, box_size, x_c, self.rcut_i, self.rcut_e, self.cn_res
        )

        self.ncn_ += 1

    def end(self, atom_type, positions, writes=True, file_cn="cn.dat"):
        """
        Parameters
        ----------
        writes : True (or False)
            if you want (or don't want) to write an output

        file_cn : filname
            the file were the cn is going to be written

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        self.cn_ : numpy array
            an array with the coordination number of each atom selected
        """

        cn_ = np.asarray(
            np.frombuffer(self.cn_res, dtype=np.intc, count=self.natoms)
        )
        self.cn_ = np.array(cn_ / self.ncn_, dtype=np.float32)

        if writes is True:
            file_cn = open(file_cn, "w")
            file_cn.write("%d\n\n" % self.natoms)

            for i in range(self.natoms):

                file_cn.write(
                    "{:d}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(
                        atom_type[i],
                        positions[i],
                        positions[self.natoms + i],
                        positions[2 * self.natoms + i],
                        self.cn_[i],
                    )
                )

            file_cn.close()

        return self.cn_


class diatomic:
    """
    cn of a diatomic system

    Parameters
    ----------
    natoms : integer
        number of atoms

    atom_type : numpy array with integers (could be char)
        type of atoms

    atom_type_a : integer (or char)
        type of central atoms

    atom_type_a : integer (or char)
        type of interacting atoms

    rcut_e : float
        external of the shell

    rcut_i : float
        internal of the shell
    """

    def __init__(
        self, natoms, atom_type, atom_type_a, atom_type_b, rcut_e, rcut_i=0.0
    ):

        self.natoms = natoms
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b
        self.rcut_e = rcut_e
        self.rcut_i = rcut_i

        self.n_a_ = np.count_nonzero(atom_type == self.atom_type_a)

        self.cn_ = np.zeros(self.n_a_, dtype=np.intc)
        self.ncn_ = 0

        self.cn_c = lib_cn.diatomic
        self.cn_c.argtypes = [
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_float,
            ct.c_float,
            ct.c_void_p,
        ]
        self.cn_res = (ct.c_int * self.n_a_)()

    def accumulate(self, box_size, atom_type, positions):
        """
        Parameters
        ----------
        box_size : numpy array with three floats
            the box size in x, y, z

        atom_type : numpy array with integers (could be char)
            type of atoms

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        box_size = box_size.astype(np.float32)
        box_size = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))

        positions = positions.astype(np.float32)
        x_c = positions.ctypes.data_as(ct.POINTER(ct.c_void_p))

        atom_type = atom_type.astype(np.intc)
        atom_c = atom_type.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.cn_c(
            self.natoms,
            box_size,
            atom_c,
            self.atom_type_a,
            self.atom_type_b,
            x_c,
            self.rcut_i,
            self.rcut_e,
            self.cn_res,
        )

        self.ncn_ += 1

    def end(self, atom_type, positions, writes=True, file_cn="cn.dat"):
        """
        Parameters
        ----------
        writes : True (or False)
            if you want (or don't want) to write an output

        file_cn : filname
            the file were the cn is going to be written

        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        self.cn_ : numpy array
            an array with the coordination number of each atom selected
        """

        cn_ = np.asarray(
            np.frombuffer(self.cn_res, dtype=np.intc, count=self.n_a_)
        )
        self.cn_ = np.array(cn_ / self.ncn_, dtype=np.float32)

        if writes is True:
            file_cn = open(file_cn, "w")
            file_cn.write("%d\n\n" % len(self.cn_))

            idx = 0
            for i in range(self.natoms):

                if atom_type[i] != self.atom_type_a:
                    continue

                file_cn.write(
                    "{:d}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(
                        atom_type[i],
                        positions[i],
                        positions[self.natoms + i],
                        positions[2 * self.natoms + i],
                        self.cn_[idx],
                    )
                )

                idx += 1

            file_cn.close()

        return self.cn_
