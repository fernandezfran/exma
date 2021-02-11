import numpy as np

class reader(object):
    """
    used to read molecular dynamics trajectory files
    """
    pass

class xyz(reader):
    """
    subclass of reader that reads xyz file
    """


    def __init__(self, file_xyz, ftype='typical'):
        """
        Parameters
        ----------
        file_xyz : file
            where the trajectories in xyz format are

        ftype : typical, property or image
            typical if is the usual xyz file
            property if in the last column there is a property
            image if in the last three columns there are the image box of the
                corresponding atom
        """
        if ((ftype != 'typical') and (ftype != 'property') and (ftype != 'image')):
            raise ValueError("ftype must be 'typical', 'property' or 'image'")
        
        self.file_xyz = open(file_xyz, "r")
        self.ftype = ftype


    def read_frame(self):
        """
        reads the actual frame of an .xyz file

        Returns
        -------
        natoms : integer
            the number of atoms in the frame
        
        atom_type : list of chars
            the type of the atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        property : numpy array (could be integer, float, char, etc)
            if ftype = 'property' was selected
        
        image : numpy array with integer data
            same as positions, if ftype = 'image' was selected 
        """
        N = self.file_xyz.readline()
        if not N: raise EOFError("There is no more frames to read")

        natoms = np.intc(N)
        self.file_xyz.readline() # usually a comment in .xyz files
        
        atom_type = [] 
        positions = np.zeros(3*natoms, dtype=np.float32)

        if (self.ftype == 'typical'):
    
            for i in range(0, natoms):

                txyz = self.file_xyz.readline().split()
                
                atom_type.append(txyz[0])

                positions[           i] = np.float32(txyz[1])
                positions[  natoms + i] = np.float32(txyz[2])
                positions[2*natoms + i] = np.float32(txyz[3])
       
            return natoms, atom_type, positions

        elif (self.ftype == 'property'):

            prop = np.zeros(natoms)

            for i in range(0, natoms):
                
                txyzp = self.file_xyz.readline().split()
                
                atom_type.append(txyzp[0])
                
                positions[           i] = np.float32(txyzp[1])
                positions[  natoms + i] = np.float32(txyzp[2])
                positions[2*natoms + i] = np.float32(txyzp[3])
                
                prop[i] = txyzp[4]

            return natoms, atom_type, positions, prop
        
        elif (self.ftype == 'image'):
            
            image = np.zeros(3*natoms, dtype=np.intc)

            for i in range(0, natoms):
                
                txyzp = self.file_xyz.readline().split()
                
                atom_type.append(txyzp[0])
                
                positions[           i] = np.float32(txyzp[1])
                positions[  natoms + i] = np.float32(txyzp[2])
                positions[2*natoms + i] = np.float32(txyzp[3])
                
                image[           i] = np.intc(txyzp[4])
                image[  natoms + i] = np.intc(txyzp[5])
                image[2*natoms + i] = np.intc(txyzp[6])

            return natoms, atom_type, positions, image


    def file_close(self):
        """
        close the file where the trajectories of the dynamics are
        """
        self.file_xyz.close()


class lammpstrj(reader):
    """
    subclass of reader that reads lammpstrj file
    """


    def __init__(self, file_lammps, ftype='custom'):
        """
        Parameters
        ----------
        file_lammps : file
            where the trajectories of lammps are

        ftype : custom, charge, image, charge_image
            custom = dump ... custom ... id type x y z
            charge = dump ... custom ... id type q x y z
            image = dump ... custom ... id type x y z ix iy iz
            charge_image = dump ... custom ... id type q x y z ix iy iz
        """
        if ((ftype != 'custom') and (ftype != 'charge') and (ftype != 'image') \
                and (ftype != 'charge_image')):
            raise ValueError("ftype must be 'custom', 'charge', 'image' or"
                             "'charge_image'")
        
        self.file_lammps = open(file_lammps, "r")
        self.ftype = ftype


    def read_frame(self):
        """
        reads the actual frame of a .lammpstrj file

        Returns
        -------
        natoms : integer
            the number of atoms in the frame
       
        box_size : numpy array
            with the box lenght in x, y, z

        atom_id : list of integers
            the id of the respective atom
        
        atom_type : list of integers
            the type of the atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        atom_q : numpy array with float32 data
            the charge of the respective atom, if ftype = 'charge' was selected
        
        image : numpy array with integer data
            same as positions, if ftype = 'image' was selected 
        """
        comment = self.file_lammps.readline()
        if not comment: raise EOFError("There is no more frames to read")
        self.file_lammps.readline()
        self.file_lammps.readline()

        natoms = np.intc(self.file_lammps.readline())
        
        self.file_lammps.readline()
      
        box_size = np.zeros(3, dtype=np.float32)
        lohi = self.file_lammps.readline().split()
        box_size[0] = np.float32(lohi[1]) - np.float32(lohi[0])    # xbox
        lohi = self.file_lammps.readline().split()
        box_size[1] = np.float32(lohi[1]) - np.float32(lohi[0])    # ybox
        lohi = self.file_lammps.readline().split()
        box_size[2] = np.float32(lohi[1]) - np.float32(lohi[0])    # zbox

        self.file_lammps.readline()

        atom_id = []
        atom_type = [] 
        positions = np.zeros(3*natoms, dtype=np.float32)

        if (self.ftype == 'custom'):
    
            for i in range(0, natoms):

                idtxyz = self.file_lammps.readline().split()
                
                atom_id.append(idtxyz[0])
                atom_type.append(idtxyz[1])

                positions[           i] = np.float32(idtxyz[2])
                positions[  natoms + i] = np.float32(idtxyz[3])
                positions[2*natoms + i] = np.float32(idtxyz[4])
       
            return natoms, box_size, atom_id, atom_type, positions

        elif (self.ftype == 'charge'):
            
            atom_q = np.zeros(natoms, dtype=np.float32)

            for i in range(0, natoms):

                idtqxyz = self.file_lammps.readline().split()
                
                atom_id.append(idtqxyz[0])
                atom_type.append(idtqxyz[1])

                positions[           i] = np.float32(idtqxyz[3])
                positions[  natoms + i] = np.float32(idtqxyz[4])
                positions[2*natoms + i] = np.float32(idtqxyz[5])
                
                atom_q[i] = np.float32(idtqxyz[2])
       
            return natoms, box_size, atom_id, atom_type, positions, atom_q

        elif (self.ftype == 'image'):

            image = np.zeros(3*natoms, dtype=np.intc)

            for i in range(0, natoms):

                idtxyzi = self.file_lammps.readline().split()
                
                atom_id.append(idtxyzi[0])
                atom_type.append(idtxyzi[1])

                positions[           i] = np.float32(idtxyzi[2])
                positions[  natoms + i] = np.float32(idtxyzi[3])
                positions[2*natoms + i] = np.float32(idtxyzi[4])

                image[         + i] = np.float32(idtxyzi[5])
                image[  natoms + i] = np.float32(idtxyzi[6])
                image[2*natoms + i] = np.float32(idtxyzi[7])
       
            return natoms, box_size, atom_id, atom_type, positions, image
        
        elif (self.ftype == 'charge_image'):
            
            atom_q = np.zeros(natoms, dtype=np.float32)
            image = np.zeros(3*natoms, dtype=np.intc)

            for i in range(0, natoms):

                idtqxyzi = self.file_lammps.readline().split()
                
                atom_id.append(idtqxyzi[0])
                atom_type.append(idtqxyzi[1])

                positions[           i] = np.float32(idtqxyzi[3])
                positions[  natoms + i] = np.float32(idtqxyzi[4])
                positions[2*natoms + i] = np.float32(idtqxyzi[5])
                
                atom_q[i] = np.float32(idtqxyzi[2])
       
                image[         + i] = np.float32(idtqxyzi[6])
                image[  natoms + i] = np.float32(idtqxyzi[7])
                image[2*natoms + i] = np.float32(idtqxyzi[8])

            return natoms, box_size, atom_id, atom_type, positions, atom_q, image


    def file_close(self):
        """
        close the file where the trajectories of the dynamics are
        """
        self.file_lammps.close()
