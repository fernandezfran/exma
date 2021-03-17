import unittest
import numpy as np

from exma import msd


class test_msd(unittest.TestCase):

    def test_monoatomic(self):
        """
        test the monoatomic mean square displacement
        """
        N = 1
        size = np.array([2.0, 2.0, 2.0])
        xi = np.array([0.0, 0.0, 0.0])

        MSD = msd.monoatomic(N, xi)
      

        reference = np.array([1.0, 12.0])
        xf = xi
        imgf = np.array([1, 1, 1])
        result = MSD.wrapped(size, xf, imgf)

        np.testing.assert_array_equal(result, reference)
        

        reference = np.array([2.0, 3.0])
        xf = np.array([1.0, 1.0, 1.0])
        result = MSD.unwrapped(xf)
        
        np.testing.assert_array_equal(result, reference)


    def test_diatomic(self):
        """
        test the diatomic mean square displacement
        """
        # two particles: one in the origin of the box (type 1) and the other int
        #   the center of the box (type 2)
        N = 2
        size = np.array([2.0, 2.0, 2.0])
        xi = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        types = np.array([1, 2]) 
        
        MSD = msd.diatomic(N, types, xi, 1, 2)

        
        reference = np.array([1.0, 3.0, 3.0, 3.0])
        xf = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        imgf = np.array([0, 1, 0, 1, 0, 1])
        result = MSD.wrapped(size, types, xf, imgf)
        
        np.testing.assert_array_equal(result, reference)


        reference = np.array([2.0, 0.0, 0.75, 0.375])
        xf = np.array([0.0, 1.5, 0.0, 1.5, 0.0, 1.5])
        result = MSD.unwrapped(types, xf)
        
        np.testing.assert_array_equal(result, reference)


if __name__ == '__main__':
    unittest.main()
