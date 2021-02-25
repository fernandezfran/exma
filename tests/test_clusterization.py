import unittest
import numpy as np

from exma import clusterization

class test_clusterization(unittest.TestCase):

    def test_dbscan(self):
        """
        test the dbscan cluster analyzer
        """
        reference = np.array([0, 0, -1])

        size = np.array([1.0, 1.0, 1.0])
        rcut = 0.2
        typ = np.array([1, 1, 1])
        x = np.array([0.0, 0.0, 0.5,
                      0.0, 0.0, 0.0,
                      0.45, 0.55, 0.0 ])

        result = clusterization.cluster(size, rcut).dbscan(typ, x, 1)

        np.testing.assert_array_equal(result[0], x)
        np.testing.assert_array_equal(result[1], reference)


if __name__ == '__main__':
    unittest.main()
