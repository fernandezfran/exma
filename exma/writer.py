import numpy as np

class writer(object):
    """
    used to write molecular dynamics trajectory files
    """

class xyz(writer):
    """
    subclass of writer that writes xyz file
    
    Parameters
    ----------
    file_xyz : filename
        name of the file where the trajectories in xyz format are going to be
        written

    ftype : typical, property or image
        
        typical if is the usual xyz file
        
        property if in the last column there is a property
        
        image if in the last three columns there are the image box of the 
        corresponding atom
    """

    def __init__(self, file_xyz, ftype='typical'):

        if ((ftype != 'typical') and (ftype != 'property') and (ftype != 'image')):
            raise ValueError("ftype must be 'typical', 'property' or 'image'")
        
        self.file_xyz = open(file_xyz, "w")
        self.ftype = ftype


    def write_frame(self, natoms, atom_type, positions, prop = [], image = []):
        """
        writes the actual frame in an .xyz file

        Parameters
        ----------
        natoms : integer
            the number of atoms in the frame
        
        atom_type : list of chars
            the type of the atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        prop : numpy array (could be integer, float, char, etc)
            if ftype = 'property' was selected
        
        image : numpy array with integer data
            same as positions, if ftype = 'image' was selected 
        """

        self.file_xyz.write("%d\n\n" % natoms)

        if (self.ftype == 'typical'):

            for i in range(0, natoms):
            
                self.file_xyz.write("{:s}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(
                    atom_type[i], positions[i], positions[natoms + i], 
                    positions[2*natoms + i]))

            return

        elif (self.ftype == 'property'):

            for i in range(0, natoms):

                self.file_xyz.write("{:s}\t{:.6e}\t{:.6e}\t{:.6e}\t{}\n".format(
                    atom_type[i], positions[i], positions[natoms + i], 
                    positions[2*natoms + i], prop[i]))

            return
        
        elif (self.ftype == 'image'):
            
            for i in range(0, natoms):

                self.file_xyz.write("{:s}\t{:.6e}\t{:.6e}\t{:.6e}\t{:d}\t{:d}\t"
                    "{:d}\n".format(atom_type[i], positions[i], 
                    positions[natoms + i], positions[2*natoms + i],
                    image[i], image[natoms + i], image[2*natoms + i]))

            return
                

    def file_close(self):
        """
        close the file where the trajectories of the dynamics were written
        """
        self.file_xyz.close()


class lammpstrj(writer):
    """
    subclass of writer that writes lammpstrj file
    
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
        
        self.file_lammps = open(file_lammps, "w")
        self.ftype = ftype
        self.timestep = 0


    def write_frame(self, natoms, box_size, atom_id, atom_type, positions, \
                    atom_q = [], image = []):
        """
        writes the actual frame in a .lammpstrj file

        Parameters
        ----------
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
        self.file_lammps.write("ITEM: TIMESTEP\n")
        self.file_lammps.write("{:d}\n".format(self.timestep))
        self.file_lammps.write("ITEM: NUMBER OF ATOMS\n")
        self.file_lammps.write("{:d}\n".format(natoms))
        self.file_lammps.write("ITEM: BOX BOUNDS pp pp pp\n")
        for i in range(0,3): self.file_lammps.write("0.0\t{:.6e}\n".format(
                                                                    box_size[i]))
        
        self.timestep += 1

        if (self.ftype == 'custom'):
    
            self.file_lammps.write("ITEM: ATOMS id type x y z\n")

            for i in range(0, natoms):

                self.file_lammps.write("{:d}\t{:d}\t{:.6e}\t{:.6e}\t{:.6e}"
                    "\n".format(atom_id[i], atom_type[i], positions[i],
                    positions[natoms + i], positions[2*natoms + i]))
       
            return

        elif (self.ftype == 'charge'):
            
            self.file_lammps.write("ITEM: ATOMS id type q x y z\n")

            for i in range(0, natoms):
                
                self.file_lammps.write("{:d}\t{:d}\t{:.6e}\t{:.6e}\t{:.6e}\t"
                    "{:.6e}\n".format(atom_id[i], atom_type[i], atom_q[i],
                    positions[i], positions[natoms + i], positions[2*natoms + i]))

            return

        elif (self.ftype == 'image'):

            self.file_lammps.write("ITEM: ATOMS id type x y z ix iy iz\n")

            for i in range(0, natoms):
                
                self.file_lammps.write("{:d}\t{:d}\t{:.6e}\t{:.6e}\t{:.6e}\t{:d}"
                    "\t{:d}\t{:d}\n".format(atom_id[i], atom_type[i], positions[i],
                    positions[natoms + i], positions[2*natoms + i],
                    image[i], image[natoms + i], image[2*natoms + i]))
       
            return
        
        elif (self.ftype == 'charge_image'):
            
            self.file_lammps.write("ITEM: ATOMS id type q x y z ix iy iz\n")

            for i in range(0, natoms):

                self.file_lammps.write("{:d}\t{:d}\t{:.6e}\t{:.6e}\t{:.6e}\t"
                    "{:.6e}\t{:d}\t{:d}\t{:d}\n".format(atom_id[i], atom_type[i],
                    atom_q[i], positions[i], positions[natoms + i], 
                    positions[2*natoms + i], image[i], image[natoms + i], 
                    image[2*natoms + i]))

            return


    def file_close(self):
        """
        close the file where the trajectories of the dynamics are
        """
        self.file_lammps.close()


class in_lammps(writer):
    """
    subclass of writer that writes a type of input file for lammps containing the
    info of the atoms

    Parameters
    ----------
    file_in : file name
        name of the file where you want to write the input info

    ftype : custom (default), charge, image, charge_image
    """

    def __init__(self, file_in, ftype='custom'):
        
        if ((ftype != 'custom') and (ftype != 'charge') and (ftype != 'image') \
                and (ftype != 'charge_image')):
            raise ValueError("ftype must be 'custom', 'charge', 'image' or"
                             "'charge_image'")
        
        self.file_in = open(file_in, "w")
        self.ftype = ftype

    def write_input(self, natoms, box_size, atom_id, atom_type, positions,
                    atom_q = [], image = []):
        """
        writes the actual frame in an input file in.
        
        Parameters
        ----------
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

        self.file_in.write("# the first three lines are comments...\n")
        self.file_in.write("# \n")
        self.file_in.write("# input file for LAMMPS generated by exma\n")

        self.file_in.write(f"{natoms:d} atoms\n")
        
        noat = np.unique(atom_type).size # number of atom types
        self.file_in.write(f"{noat:d} atom types\n\n")
        
        self.file_in.write(f"0.0 \t {box_size[0]:.6e} \t xlo xhi\n")
        self.file_in.write(f"0.0 \t {box_size[1]:.6e} \t ylo yhi\n")
        self.file_in.write(f"0.0 \t {box_size[2]:.6e} \t zlo zhi\n\n")

        self.file_in.write("Atoms\n\n")

        if (self.ftype == 'custom'):
            
            for i in range(natoms):
                self.file_in.write("{:d} {:d} {:.6e} {:.6e} {:.6e}\n".format(
                    atom_id[i], atom_type[i], positions[i], positions[natoms + i],
                    positions[2*natoms + i])
                    )

            return

        elif (self.ftype == 'charge'):
        
            for i in range(natoms):
                self.file_in.write("{:d} {:d} {:.6e} {:.6e} {:.6e} "
                    "{:.6e}\n".format(atom_id[i], atom_type[i], atom_q[i],
                    positions[i], positions[natoms + i],
                    positions[2*natoms + i])
                    )

            return
        
        elif (self.ftype == 'image'):

            for i in range(natoms):
                self.file_in.write("{:d} {:d} {:.6e} {:.6e} {:.6e} {:d} {:d}"
                    "{:d}\n".format(atom_id[i], atom_type[i], positions[i],
                    positions[natoms + i], positions[2*natoms + i], image[i],
                    image[natoms + i], image[2*natoms + i])
                    )

            return

        elif (self.ftype == 'charge_image'):
            
            for i in range(natoms):
                self.file_in.write("{:d} {:d} {:.6e} {:.6e} {:.6e} {:.6e} {:d}"
                    "{:d} {:d}\n".format(atom_id[i], atom_type[i], positions[i],
                    positions[natoms + i], positions[2*natoms + i], atom_q[i],
                    image[i], image[natoms + i], image[2*natoms + i])
                    )

            return

        self.file_in.close()
