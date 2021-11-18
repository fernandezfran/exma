import unittest

import numpy as np

from exma import msd


class TestMSD(unittest.TestCase):
    def test_monoatomic(self):
        """
        test the monoatomic mean square displacement
        """
        natoms = 1
        size = np.array([2.0, 2.0, 2.0])
        xi = np.array([0.0, 0.0, 0.0])

        meansd = msd.monoatomic(natoms, size, xi)

        reference = np.array([1.0, 12.0])
        xf = xi
        imgf = np.array([1, 1, 1])
        result = meansd.wrapped(size, xf, imgf)

        np.testing.assert_array_equal(result, reference)

        reference = np.array([2.0, 3.0])
        xf = np.array([1.0, 1.0, 1.0])
        result = meansd.unwrapped(xf)

        np.testing.assert_array_equal(result, reference)

    def test_diatomic(self):
        """
        test the diatomic mean square displacement
        """
        # two particles: one in the origin of the box (type 1) and the other
        # int the center of the box (type 2)
        natoms = 2
        size = np.array([2.0, 2.0, 2.0])
        xi = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        types = np.array([1, 2])

        meansd = msd.diatomic(natoms, size, types, xi, 1, 2)

        reference = np.array([1.0, 3.0, 3.0, 3.0])
        xf = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        imgf = np.array([0, 1, 0, 1, 0, 1])
        result = meansd.wrapped(size, types, xf, imgf)

        np.testing.assert_array_equal(result, reference)

        reference = np.array([2.0, 0.0, 0.75, 0.375])
        xf = np.array([0.0, 1.5, 0.0, 1.5, 0.0, 1.5])
        result = meansd.unwrapped(types, xf)

        np.testing.assert_array_equal(result, reference)


if __name__ == "__main__":
    unittest.main()
