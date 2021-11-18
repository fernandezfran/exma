import unittest

import numpy as np

from exma import reader, writer


class TestWriterAndReader(unittest.TestCase):
    def test_xyz(self):
        """
        write and read .xyz file
        """
        natoms = 5
        types = ["H"] * natoms
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )

        wxyz = writer.xyz("tests/data/test.xyz")
        wxyz.write_frame(natoms, types, x)
        wxyz.file_close()

        rxyz = reader.xyz("tests/data/test.xyz")
        result = rxyz.read_frame()
        rxyz.file_close()

        self.assertEqual(result[0], natoms)
        self.assertListEqual(result[1], types)
        np.testing.assert_array_almost_equal(result[2], x)

    def test_xyz_property(self):
        """
        write and read .xyz file with a property in the last column
        """
        natoms = 5
        types = ["H"] * natoms
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )
        prop = np.arange(0, 5)

        wxyz = writer.xyz("tests/data/test.xyz", "property")
        wxyz.write_frame(natoms, types, x, prop)
        wxyz.file_close()

        rxyz = reader.xyz("tests/data/test.xyz", "property")
        result = rxyz.read_frame()
        rxyz.file_close()

        self.assertEqual(result[0], natoms)
        self.assertListEqual(result[1], types)
        np.testing.assert_array_almost_equal(result[2], x)
        np.testing.assert_array_almost_equal(result[3], prop)

    def test_xyz_image(self):
        """
        write and read .xyz file with the corresponding images in the last
        column
        """
        natoms = 5
        types = ["H"] * natoms
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )
        img = np.array([0, 0, 1, -1, 0, 2, 3, 0, 0, 1, -2, -1, 0, 0, 1])

        wxyz = writer.xyz("tests/data/test.xyz", "image")
        wxyz.write_frame(natoms, types, x, image=img)
        wxyz.file_close()

        rxyz = reader.xyz("tests/data/test.xyz", "image")
        result = rxyz.read_frame()
        rxyz.file_close()

        self.assertEqual(result[0], natoms)
        self.assertListEqual(result[1], types)
        np.testing.assert_array_almost_equal(result[2], x)
        np.testing.assert_array_almost_equal(result[3], img)

    def test_xyz_error(self):
        """
        check raises write and read .xyz file
        """
        with self.assertRaises(ValueError):
            writer.xyz("tests/data/test.xyz", "error")

        with self.assertRaises(ValueError):
            reader.xyz("tests/data/test.xyz", "error")

        rxyz = reader.xyz("tests/data/test.xyz")
        rxyz.read_frame()
        with self.assertRaises(EOFError):
            rxyz.read_frame()
        rxyz.file_close()

    def test_lammpstrj(self):
        """
        write and read .lammpstrj file
        """
        natoms = 5
        size = np.array([4.5, 1.0, 6.0])
        idx = np.arange(1, natoms + 1)
        types = np.array([1, 1, 1, 2, 2])
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )

        wlmp = writer.lammpstrj("tests/data/test.lammpstrj")
        wlmp.write_frame(natoms, size, idx, types, x)
        wlmp.file_close()

        rlmp = reader.lammpstrj("tests/data/test.lammpstrj")
        result = rlmp.read_frame()
        rlmp.file_close()

        self.assertEqual(result[0], natoms)
        np.testing.assert_array_almost_equal(result[1], size)
        np.testing.assert_array_equal(result[2], idx)
        np.testing.assert_array_equal(result[3], types)
        np.testing.assert_array_almost_equal(result[4], x)

    def test_lammpstrj_charge(self):
        """
        write and read .lammpstrj file with charges
        """
        natoms = 5
        size = np.array([4.5, 1.0, 6.0])
        idx = np.arange(1, natoms + 1)
        types = np.array([1, 1, 1, 2, 2])
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )
        q = np.array([-0.3356, -0.32636, -0.34256, 0.54365, 0.46463])

        wlmp = writer.lammpstrj("tests/data/test.lammpstrj", "charge")
        wlmp.write_frame(natoms, size, idx, types, x, q)
        wlmp.file_close()

        rlmp = reader.lammpstrj("tests/data/test.lammpstrj", "charge")
        result = rlmp.read_frame()
        rlmp.file_close()

        self.assertEqual(result[0], natoms)
        np.testing.assert_array_almost_equal(result[1], size)
        np.testing.assert_array_equal(result[2], idx)
        np.testing.assert_array_equal(result[3], types)
        np.testing.assert_array_almost_equal(result[4], x)
        np.testing.assert_array_almost_equal(result[5], q)

    def test_lammpstrj_image(self):
        """
        write and read .lammpstrj file with images box
        """
        natoms = 5
        size = np.array([4.5, 1.0, 6.0])
        idx = np.arange(1, natoms + 1)
        types = np.array([1, 1, 1, 2, 2])
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )
        img = np.array([0, 0, 1, -1, 0, 2, 3, 0, 0, 1, -2, -1, 0, 0, 1])

        wlmp = writer.lammpstrj("tests/data/test.lammpstrj", "image")
        wlmp.write_frame(natoms, size, idx, types, x, image=img)
        wlmp.file_close()

        rlmp = reader.lammpstrj("tests/data/test.lammpstrj", "image")
        result = rlmp.read_frame()
        rlmp.file_close()

        self.assertEqual(result[0], natoms)
        np.testing.assert_array_almost_equal(result[1], size)
        np.testing.assert_array_equal(result[2], idx)
        np.testing.assert_array_equal(result[3], types)
        np.testing.assert_array_almost_equal(result[4], x)
        np.testing.assert_array_equal(result[5], img)

    def test_lammpstrj_charge_image(self):
        """
        write and read .lammpstrj file with images box and charges
        """
        natoms = 5
        size = np.array([4.5, 1.0, 6.0])
        idx = np.arange(1, natoms + 1)
        types = np.array([1, 1, 1, 2, 2])
        x = np.array(
            [
                2.67583,
                0.93241,
                1.23424,
                4.42636,
                3.00023,
                0.05432,
                0.89325,
                0.43142,
                0.23451,
                0.55556,
                1.15145,
                2.31451,
                3.96893,
                4.96905,
                5.98693,
            ]
        )
        q = np.array([-0.3356, -0.32636, -0.34256, 0.54365, 0.46463])
        img = np.array([0, 0, 1, -1, 0, 2, 3, 0, 0, 1, -2, -1, 0, 0, 1])

        wlmp = writer.lammpstrj("tests/data/test.lammpstrj", "charge_image")
        wlmp.write_frame(natoms, size, idx, types, x, q, img)
        wlmp.file_close()

        rlmp = reader.lammpstrj("tests/data/test.lammpstrj", "charge_image")
        result = rlmp.read_frame()
        rlmp.file_close()

        self.assertEqual(result[0], natoms)
        np.testing.assert_array_almost_equal(result[1], size)
        np.testing.assert_array_equal(result[2], idx)
        np.testing.assert_array_equal(result[3], types)
        np.testing.assert_array_almost_equal(result[4], x)
        np.testing.assert_array_almost_equal(result[5], q)
        np.testing.assert_array_equal(result[6], img)

    def test_lammpstrj_error(self):
        """
        check raises write and read .lammpstrj file
        """
        with self.assertRaises(ValueError):
            writer.xyz("tests/data/test.lammpstrj", "error")

        with self.assertRaises(ValueError):
            reader.xyz("tests/data/test.lammpstrj", "error")

        rlmp = reader.lammpstrj("tests/data/test.lammpstrj")
        rlmp.read_frame()
        with self.assertRaises(EOFError):
            rlmp.read_frame()
        rlmp.file_close()


if __name__ == "__main__":
    unittest.main()
