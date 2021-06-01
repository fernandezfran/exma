import unittest
import numpy as np

from exma import atoms

class test_positions(unittest.TestCase):

    def test_sc(self):
        """
        test that the atoms are placed in a simple cubic crystal
        """
        reference = np.array([
            0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5,
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5
            ])    # it has x in the first line, y in the second and z in the third

        particles = atoms.positions(8, 1.0)
        result = particles.sc()
        
        np.testing.assert_array_equal(result, reference)
    

    def test_bcc(self):
        """
        test that the atoms are placed in a body-centered cubic crystal
        """
        reference = np.array([
            0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
            0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75,
            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5,
            0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75,
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
            0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75
            ])
        
        particles = atoms.positions(16, 1.0)
        result = particles.bcc()
        
        np.testing.assert_array_equal(result, reference)


    def test_fcc(self):
        """
        test that the atoms are placed in a face-centered cubic crystal
        """
        reference = np.array([
            0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
            0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75,
            0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75,
            0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 
            0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75,
            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5,
            0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75,
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
            0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75,
            0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75
            ])

        particles = atoms.positions(32, 1.0)
        result = particles.fcc()
        
        np.testing.assert_array_equal(result, reference)
    
    
    def test_dcc(self):
        """
        test that the atoms are placed in a diamond cubic crystal
        """
        reference = np.array([ 0.25, 0.00, 0.25, 0.00, 0.75, 0.50, 0.75, 0.50,
                               0.75, 0.00, 0.25, 0.50, 0.75, 0.00, 0.25, 0.50,
                               0.25, 0.50, 0.75, 0.00, 0.75, 0.00, 0.25, 0.50 ])

        particles = atoms.positions(8, 1.0)
        result = particles.dcc()
        
        np.testing.assert_array_equal(result, reference)

    
    def test_error_sc(self):
        """
        test error of the simple cubic crystal (when number of atoms is not a 
        power of three)
        """
        particles = atoms.positions(7, 1.0)
        with self.assertRaises(ValueError):
            result = particles.sc()
    

    def test_error_bcc(self):
        """
        test error of the body-centered cubic crystal (when number of atoms is not
        a power of three multiply by two)
        """
        particles = atoms.positions(19, 1.0)
        with self.assertRaises(ValueError):
            result = particles.bcc()


    def test_error_fcc(self):
        """
        test error of the face-centered cubic crystal (when number of atoms is not
        a power of three multiply by two)
        """
        particles = atoms.positions(37, 1.0)
        with self.assertRaises(ValueError):
            result = particles.fcc()


    def test_error_fcc(self):
        """
        test error of the diamond cubic crystal (when number of atoms is invalid)
        """
        particles = atoms.positions(7, 1.0)
        with self.assertRaises(ValueError):
            result = particles.dcc()


if __name__ == '__main__':
    unittest.main()
