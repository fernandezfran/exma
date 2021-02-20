import unittest
import numpy as np

from exma import boundary


class test_apply(unittest.TestCase):

    def test_pbc(self):
        """
        test that the periodic boundary conditions work fine
        """
        reference = np.array([0.5, 0.5, 0.0,
                              1.0, 1.0, 0.0,
                              0.2, 0.8, 0.5], dtype=np.float32)
  
        PBC = boundary.apply(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        result = PBC.pbc(3, np.array([-0.5, 0.5, 0.0,
                                       1.0, 3.0, -1.0,
                                       0.2, 0.8, 0.5], dtype=np.float32))
        
        np.testing.assert_array_almost_equal(result, reference)


    def test_minimum_image(self):
        """
        test that minimum image work fine
        """
        reference = np.array([0.4, 0.0, -0.2], dtype=np.float32)

        minimg = boundary.apply(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        x = np.array([0.3, 0.5, 0.1])
        y = np.array([0.7, 0.5, 0.9])
        result = minimg.minimum_image(x, y)

        np.testing.assert_array_almost_equal(result, reference)


if __name__ == '__main__':
    unittest.main()
