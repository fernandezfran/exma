import unittest
import numpy as np

from exma import statistics


class test_statistics(unittest.TestCase):

    def test_block_average(self):
        """
        test the estimation of an error
        """
        reference = 0

        BA = statistics.block_average(file_data="tests/data/test_block_average.dat")
        result = BA.estimate_error()

        np.testing.assert_array_equal(result[0], np.array([0, 1]))
        np.testing.assert_array_equal(result[1], np.array([8, 4]))
        np.testing.assert_array_almost_equal(
                          result[2], np.array([3.1412501, 3.1412501])
                          )
        np.testing.assert_array_almost_equal(
                          result[3], np.array([2.299121e-05, 2.656272e-05])
                          )
        np.testing.assert_array_almost_equal(
                          result[4], np.array([1.228932e-05, 2.168837e-05])
                          )


if __name__ == '__main__':
    unittest.main()
