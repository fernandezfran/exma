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


    def test_error_dcc(self):
        """
        test error of the diamond cubic crystal (when number of atoms is invalid)
        """
        particles = atoms.positions(9, 1.0)
        with self.assertRaises(ValueError):
            result = particles.dcc()


class test_nanoparticle(unittest.TestCase):

    def test_snp(self):
        """
        test the spherical nanoparticle
        """
        reference = np.array([-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5, 
                               0.0, -0.5,  0.0,  0.0,  0.0,  0.5,  0.0, 
                               0.0,  0.0, -0.5,  0.0,  0.5,  0.0,  0.0])

        NP = atoms.nanoparticle(atoms.positions(8, 1.0).sc(), np.full(3, 1.0))
        result = NP.spherical(0.6)

        np.testing.assert_array_equal(result, reference)


class test_replicate(unittest.TestCase):

    def test_crystal(self):
        """
        silicon diamond example
        """
        N_reference = 8 * 2 * 2 * 2
        box_reference = np.full(3, 2 * 5.468728)
        typ_reference = ['Si'] * 8 * 2 * 2 * 2
        pos_reference = np.array(
                [1.367182, 0.      , 1.367182, 0.      , 4.101546, 2.734364,
                 4.101546, 2.734364, 1.367182, 0.      , 1.367182, 0.      ,
                 4.101546, 2.734364, 4.101546, 2.734364, 1.367182, 0.      ,
                 1.367182, 0.      , 4.101546, 2.734364, 4.101546, 2.734364,
                 1.367182, 0.      , 1.367182, 0.      , 4.101546, 2.734364,
                 4.101546, 2.734364, 6.83591 , 5.468728, 6.83591 , 5.468728,
                 9.570274, 8.203092, 9.570274, 8.203092, 6.83591 , 5.468728,
                 6.83591 , 5.468728, 9.570274, 8.203092, 9.570274, 8.203092,
                 6.83591 , 5.468728, 6.83591 , 5.468728, 9.570274, 8.203092,
                 9.570274, 8.203092, 6.83591 , 5.468728, 6.83591 , 5.468728,
                 9.570274, 8.203092, 9.570274, 8.203092, 4.101546, 0.      ,
                 1.367182, 2.734364, 4.101546, 0.      , 1.367182, 2.734364,
                 4.101546, 0.      , 1.367182, 2.734364, 4.101546, 0.      ,
                 1.367182, 2.734364, 9.570274, 5.468728, 6.83591 , 8.203092,
                 9.570274, 5.468728, 6.83591 , 8.203092, 9.570274, 5.468728,
                 6.83591 , 8.203092, 9.570274, 5.468728, 6.83591 , 8.203092,
                 4.101546, 0.      , 1.367182, 2.734364, 4.101546, 0.      ,
                 1.367182, 2.734364, 4.101546, 0.      , 1.367182, 2.734364,
                 4.101546, 0.      , 1.367182, 2.734364, 9.570274, 5.468728,
                 6.83591 , 8.203092, 9.570274, 5.468728, 6.83591 , 8.203092,
                 9.570274, 5.468728, 6.83591 , 8.203092, 9.570274, 5.468728,
                 6.83591 , 8.203092, 1.367182, 2.734364, 4.101546, 0.      ,
                 4.101546, 0.      , 1.367182, 2.734364, 6.83591 , 8.203092,
                 9.570274, 5.468728, 9.570274, 5.468728, 6.83591 , 8.203092,
                 1.367182, 2.734364, 4.101546, 0.      , 4.101546, 0.      ,
                 1.367182, 2.734364, 6.83591 , 8.203092, 9.570274, 5.468728,
                 9.570274, 5.468728, 6.83591 , 8.203092, 1.367182, 2.734364,
                 4.101546, 0.      , 4.101546, 0.      , 1.367182, 2.734364,
                 6.83591 , 8.203092, 9.570274, 5.468728, 9.570274, 5.468728,
                 6.83591 , 8.203092, 1.367182, 2.734364, 4.101546, 0.      ,
                 4.101546, 0.      , 1.367182, 2.734364, 6.83591 , 8.203092,
                 9.570274, 5.468728, 9.570274, 5.468728, 6.83591 , 8.203092])

        
        rep = atoms.replicate(8, np.full(3, 5.468728), ['Si'] * 8, atoms.positions(8, 1.0).dcc())
        result = rep.crystal(2, 2, 2)

        np.testing.assert_array_equal(result[0], N_reference)
        np.testing.assert_array_equal(result[1], box_reference)
        np.testing.assert_array_equal(result[2], typ_reference)
        np.testing.assert_array_almost_equal(result[3], pos_reference)


if __name__ == '__main__':
    unittest.main()
