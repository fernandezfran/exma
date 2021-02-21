import unittest
import numpy as np

from exma import statistics


class test_statistics(unittest.TestCase):

    def test_block_average(self):
        """
        test the estimation of an error
        """
        reference = 0

        BA = statistics.block_average("tests/data/test_block_average.dat", 0)
        result = BA.estimate_error()

        self.assertListEqual(result[0], [0, 1])
        self.assertListEqual(result[1], [8, 4])
        np.testing.assert_array_almost_equal(result[2], [3.1412501, 3.1412501])
        np.testing.assert_array_almost_equal(result[3], [2.299121e-05, 2.656272e-05])
        np.testing.assert_array_almost_equal(result[4], [1.228932e-05, 2.168837e-05])


if __name__ == '__main__':
    unittest.main()
