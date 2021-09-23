import unittest

import numpy as np

from exma import sro


class TestSRO(unittest.TestCase):
    def test_amorphous(self):
        """
        test the amorphous parameter calculation in the rdf of a fcc crystal
        """
        reference = -0.8731494

        rdf_x = np.arange(0.025, 0.5, 0.05)
        rdf_y = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                3.478797,
                0.0,
                0.835459,
                0.0,
                1.955821,
                0.78305,
            ]
        )
        rcut = 0.375

        shortro = sro.amorphous(rdf_x, rdf_y)
        result = shortro.parameter(rcut)

        self.assertAlmostEqual(result, reference)


if __name__ == "__main__":
    unittest.main()
