import unittest
import numpy as np

from exma.EN import en


class test_en(unittest.TestCase):

    def test_effective_neighbors(self):
        """
        test the calculation of effective_neighbors
        """
        reference = np.array([1.0, 2.0])

        # this is a dumbell of atoms type 1 in y-direction crossed by a dumbell 
        #   of atoms type 2 in z-direction and an isolated atom near second atom
        #   of type 1. then, the first atom of type 1 has 1 effective neighbor
        #   (half of each dumbell of type 2), the same for the second atom plus
        #   the isolated atom, so it has 2 effective neighbor.
        N = 5
        size = np.array([1.0, 1.0, 1.0])
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5,
                      0.4, 0.6, 0.5, 0.5, 0.7,
                      0.5, 0.5, 0.4, 0.6, 0.5])
        types = np.array([1, 1, 2, 2, 2])

        effnei = en.effective_neighbors(N, 1, 2)
        result = effnei.of_this_frame(size, types, x)

        np.testing.assert_array_almost_equal(result, reference, 5)


if __name__ == '__main__':
    unittest.main()
