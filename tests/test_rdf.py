import unittest
import numpy as np
from exma import atoms

from exma.RDF import rdf

class test_rdf(unittest.TestCase):

    def test_monoatomic(self):
        """
        test the radial distribution function of a monoatomic face-centered
        cubic crystal
        """
        reference_x = np.arange(0.025, 0.5, 0.05)
        reference_y = np.array([0.0, 0.0 , 0.0 , 0.0, 3.478797, 0.0, 0.835459, 
                                0.0, 1.955821, 0.78305])
        reference = np.split(np.concatenate((reference_x, reference_y)), 2)

        N = 108
        size = np.array([1.0, 1.0, 1.0])

        particles = atoms.positions(N, size[0])
        x = particles.fcc()

        gofr = rdf.monoatomic(N, size, 10)
        gofr.accumulate(x)
        result = gofr.end(False)
        
        np.testing.assert_array_almost_equal(result, reference)

    def test_diatomic(self):
        """
        test the radial distribution function of a diatomic body-centered
        cubic crystal
        """
        reference_x = np.arange(0.025, 0.5, 0.05)
        reference_y = np.array([0.0 , 0.0 , 0.0, 0.0, 0.0, 6.218508, 0.0 , 0.0, 
                                0.0, 0.0])
        reference = np.split(np.concatenate((reference_x, reference_y)), 2)

        N = 54
        size = np.array([1.0,1.0,1.0])
        
        type1 = np.full(np.intc(N/2), 1)
        type2 = np.full(np.intc(N/2), 2)
        types = np.concatenate((type1, type2))
        particles = atoms.positions(N, size[0])
        x = particles.bcc()

        gofr = rdf.diatomic(N, size, 10, 1, 2)
        gofr.accumulate(types, x)
        result = gofr.end(types, False)

        np.testing.assert_array_almost_equal(result, reference)

if __name__ == '__main__':
    unittest.main()
