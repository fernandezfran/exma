import unittest
import numpy as np
from exma import atoms

from exma import sro


class test_sro(unittest.TestCase):

    def test_warren_cowley(self):
        """
        test the calculation of the short range order warren cowley parameter
        """
        reference = np.array([0.0, 0.0])

        N = 54
        size = np.array([1.0, 1.0, 1.0])
        rcut = 0.75

        type1 = np.full(np.intc(N/2), 1)
        type2 = np.full(np.intc(N/2), 2)
        types = np.concatenate((type1, type2))
        particles = atoms.positions(N, size[0])
        x = particles.bcc()
        
        WCP = sro.warren_cowley(N, size, types, 1, 2, rcut)
        WCP.accumulate(types, x)
        result = WCP.end()

        np.testing.assert_array_equal(result, reference)


if __name__ == '__main__':
    unittest.main()
