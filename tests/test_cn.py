import unittest

import numpy as np

from exma import atoms
from exma.cn.coordination_number import diatomic, monoatomic


class TestCN(unittest.TestCase):
    def test_monoatomic(self):
        """
        test the coordination number of a monoatomic simple cubic crystal
        """
        natoms = 27
        size = np.array([1.0, 1.0, 1.0])
        rcut = 0.4

        reference = np.full(natoms, 6.0)

        particles = atoms.positions(natoms, size[0])
        x = particles.sc()

        mono = monoatomic(natoms, rcut)
        mono.accumulate(size, x)
        result = mono.end(0, x, writes=False)

        np.testing.assert_array_equal(result, reference)

    def test_diatomic(self):
        """
        test the coordination number of diatomic body-centered cubic crystal
        """
        natoms = 54
        size = np.array([1.0, 1.0, 1.0])
        rcut = 0.3

        reference = np.full(np.intc(natoms / 2), 8.0)

        type1 = np.full(np.intc(natoms / 2), 1)
        type2 = np.full(np.intc(natoms / 2), 2)
        types = np.concatenate((type1, type2))
        particles = atoms.positions(natoms, size[0])
        x = particles.bcc()

        di = diatomic(natoms, types, 1, 2, rcut)
        di.accumulate(size, types, x)
        result = di.end(types, x, writes=False)

        np.testing.assert_array_equal(result, reference)


if __name__ == "__main__":
    unittest.main()
