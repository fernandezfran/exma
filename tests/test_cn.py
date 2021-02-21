import unittest
import numpy as np
from exma import atoms

from exma import cn


class test_cn(unittest.TestCase):

    def test_monoatomic(self):
        """
        test the coordination number of a monoatomic simple cubic crystal
        """
        N = 27
        size = np.array([1.0, 1.0, 1.0])
        rcut = 0.4
        
        reference = np.full(N, 6.0)
        
        particles = atoms.positions(N, size[0])
        x = particles.sc()

        mono = cn.monoatomic(N, size, rcut)
        mono.accumulate(x)
        result = mono.end(0, x, writes=False)

        np.testing.assert_array_equal(result, reference)


    def test_diatomic(self):
        """
        test the coordination number of diatomic body-centered cubic crystal
        """
        N = 54
        size = np.array([1.0, 1.0, 1.0])
        rcut = 0.3
        
        reference = np.full(np.intc(N/2), 8.0)

        type1 = np.full(np.intc(N/2), 1)
        type2 = np.full(np.intc(N/2), 2)
        types = np.concatenate((type1, type2))
        particles = atoms.positions(N, size[0])
        x = particles.bcc()
        
        di = cn.diatomic(N, size, types, 1, 2, rcut)
        di.accumulate(types, x)
        result = di.end(types, x, writes=False)
        
        np.testing.assert_array_equal(result, reference)


if __name__ == '__main__':
    unittest.main()
