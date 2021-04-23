import numpy as np

class reader(object):
    """
    used to read molecular dynamics trajectory files
    """

class xyz(reader):
    """
    subclass of reader that reads xyz file
 
    Parameters
    ----------
    file_xyz : filename
        name of the file where the trajectories in xyz format are

    ftype : typical, property or image

        typical if is the usual xyz file
        
        property if in the last column there is a property
        
        image if in the last three columns there are the image box of the
        corresponding atom
    """

    def __init__(self, file_xyz, ftype='typical'):

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

        if (self.ftype == 'typical'):
            
            x, y, z = [], [], []
            for i in range(0, natoms):

                txyz = self.file_xyz.readline().split()
                
                atom_type.append(txyz[0])

                x.append(txyz[1])
                y.append(txyz[2])
                z.append(txyz[3])

            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)

            positions = np.concatenate((x,y,z))
       
            return natoms, atom_type, positions

        elif (self.ftype == 'property'):

            x, y, z, prop = [], [], [], []
            for i in range(0, natoms):
                
                txyzp = self.file_xyz.readline().split()
                
                atom_type.append(txyzp[0])
                
                x.append(txyzp[1])
                y.append(txyzp[2])
                z.append(txyzp[3])

                prop.append(txyzp[4])

            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)
            
            positions = np.concatenate((x,y,z))
            
            prop = np.asarray(prop, dtype=np.float32)

            return natoms, atom_type, positions, prop
        
        elif (self.ftype == 'image'):

            x, y, z = [], [], []
            ix, iy, iz = [], [], []
            for i in range(0, natoms):
                
                txyzi = self.file_xyz.readline().split()
                
                atom_type.append(txyzi[0])
                
                x.append(txyzi[1])
                y.append(txyzi[2])
                z.append(txyzi[3])
                
                ix.append(txyzi[4])
                iy.append(txyzi[5])
                iz.append(txyzi[6])

            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)

            ix = np.asarray(ix, dtype=np.intc)
            iy = np.asarray(iy, dtype=np.intc)
            iz = np.asarray(iz, dtype=np.intc)
                
            positions = np.concatenate((x,y,z))
            image = np.concatenate((ix,iy,iz))

            return natoms, atom_type, positions, image


    def file_close(self):
        """
        close the file where the trajectories of the dynamics are
        """
        self.file_xyz.close()


class lammpstrj(reader):
    """
    subclass of reader that reads lammpstrj file
    
    Parameters
    ----------
    file_lammps : filename
        name of the file where the trajectories of lammps are

    ftype : custom, charge, image, charge_image
        
        custom = dump ... custom ... id type x y z
        
        charge = dump ... custom ... id type q x y z
        
        image = dump ... custom ... id type x y z ix iy iz
        
        charge_image = dump ... custom ... id type q x y z ix iy iz
    """

    def __init__(self, file_lammps, ftype='custom'):
        
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

        if (self.ftype == 'custom'):
            
            x, y, z = [], [], []
            for i in range(0, natoms):

                idtxyz = self.file_lammps.readline().split()
                
                atom_id.append(idtxyz[0])
                atom_type.append(idtxyz[1])

                x.append(idtxyz[2])
                y.append(idtxyz[3])
                z.append(idtxyz[4])
            
            atom_id = np.array(atom_id, dtype=np.intc)
            atom_type = np.array(atom_type, dtype=np.intc)
            
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)

            positions = np.concatenate((x,y,z))

            return natoms, box_size, atom_id, atom_type, positions

        elif (self.ftype == 'charge'):

            q, x, y, z = [], [], [], []

            for i in range(0, natoms):
              
                idtqxyz = self.file_lammps.readline().split()
                
                atom_id.append(idtqxyz[0])
                atom_type.append(idtqxyz[1])

                q.append(idtqxyz[2])
               
                x.append(idtqxyz[3])
                y.append(idtqxyz[4])
                z.append(idtqxyz[5])

            atom_id = np.array(atom_id, dtype=np.intc)
            atom_type = np.array(atom_type, dtype=np.intc)
            
            atom_q = np.asarray(q, dtype=np.float32)
            
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)

            positions = np.concatenate((x,y,z))

            return natoms, box_size, atom_id, atom_type, positions, atom_q

        elif (self.ftype == 'image'):

            x, y, z = [], [], []
            ix, iy, iz = [], [], []
            for i in range(0, natoms):

                idtxyzi = self.file_lammps.readline().split()
                
                atom_id.append(idtxyzi[0])
                atom_type.append(idtxyzi[1])
                
                x.append(idtxyzi[2])
                y.append(idtxyzi[3])
                z.append(idtxyzi[4])
                
                ix.append(idtxyzi[5])
                iy.append(idtxyzi[6])
                iz.append(idtxyzi[7])
       
            atom_id = np.array(atom_id, dtype=np.intc)
            atom_type = np.array(atom_type, dtype=np.intc)
            
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)
            
            ix = np.asarray(ix, dtype=np.intc)
            iy = np.asarray(iy, dtype=np.intc)
            iz = np.asarray(iz, dtype=np.intc)

            positions = np.concatenate((x,y,z))
            image = np.concatenate((ix,iy,iz))

            return natoms, box_size, atom_id, atom_type, positions, image
        
        elif (self.ftype == 'charge_image'):
            
            atom_q = np.zeros(natoms, dtype=np.float32)
            image = np.zeros(3*natoms, dtype=np.intc)

            q, x, y, z = [], [], [], []
            ix, iy, iz = [], [], []
            for i in range(0, natoms):

                idtqxyzi = self.file_lammps.readline().split()
                
                atom_id.append(idtqxyzi[0])
                atom_type.append(idtqxyzi[1])
                q.append(idtqxyzi[2])
               
                x.append(idtqxyzi[3])
                y.append(idtqxyzi[4])
                z.append(idtqxyzi[5])
                
                ix.append(idtqxyzi[6])
                iy.append(idtqxyzi[7])
                iz.append(idtqxyzi[8])


            atom_id = np.array(atom_id, dtype=np.intc)
            atom_type = np.array(atom_type, dtype=np.intc)
            atom_q = np.asarray(q, dtype=np.float32)
            
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)
            
            ix = np.asarray(ix, dtype=np.intc)
            iy = np.asarray(iy, dtype=np.intc)
            iz = np.asarray(iz, dtype=np.intc)

            positions = np.concatenate((x,y,z))
            image = np.concatenate((ix,iy,iz))
            
            return natoms, box_size, atom_id, atom_type, positions, atom_q, image


    def file_close(self):
        """
        close the file where the trajectories of the dynamics are
        """
        self.file_lammps.close()
