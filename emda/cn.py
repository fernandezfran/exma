import numpy as np
import boundary

class cn:
    """
    coordination number
    """

class monoatomic(cn):
    """
    cn of a monoatomic system
    """
    
    def __init__(self, natoms, box_size, rcut_e, rcut_i=0.0):
        """
        Parameters
        ----------
        natoms : integer
            number of atoms

        box_size : numpy array with three floats
            the box size in x, y, z

        rcut_e : float
            external of the shell

        rcut_i : float
            internal of the shell
        """
        self.natoms = natoms
        self.box_size = box_size
        self.rcut_e = rcut_e
        self.rcut_i = rcut_i

        self.cn = np.zeros(self.natoms)
        self.ncn = 0
        self.bound = boundary.apply(self.box_size)
        

    def accumulate(self, positions):
        """
        Parameters
        ----------
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        for i in range(0, self.natoms-1):

            ri = [positions[i + k*self.natoms] for k in range(0,3)]
            
            for j in range(i+1, self.natoms):

                rj = [positions[j + k*self.natoms] for k in range(0,3)]

                rij = self.bound.minimum_image(ri, rj)
                rij = np.linalg.norm(rij)

                if (rij <= self.rcut_i or rij > self.rcut_e): continue
                self.cn[idx] += 1

        self.ncn +=1
    

    def end(self, atom_type, positions, writes=True, file_cn='cn.dat'):
        """
        Parameters
        ----------
        writes : True (or False)
            if you want (or don't want) to write an output

        file_cn : filname
            the file were the cn is going to be written
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        self.cn : numpy array
            an array with the coordination number of each atom selected
        """

        if (writes == True):
            file_cn = open(file_cn, 'w')
            file_cn.write("%d\n\n" % self.natoms)

            for i in range(0, self.natoms):
            
                file_cn.write("%d %g %g %g %g\n" % (atom_type[i], positions[i], \
                    positions[self.natoms + i], positions[2*self.natoms + i], \
                    self.cn[i]/self.ncn))

            file_cn.close()

        return np.array(self.cn / self.ncn, dtype=np.float32)


class diatomic(cn):
    """
    cn of a diatomic system
    """
    
    def __init__(self, natoms, box_size, atom_type, atom_type_a, atom_type_b, \
                    rcut_e, rcut_i=0.0):
        """
        Parameters
        ----------
        natoms : integer
            number of atoms

        box_size : numpy array with three floats
            the box size in x, y, z

        atom_type : numpy array with integers (could be char)
            type of atoms
        
        atom_type_a : integer (or char)
            type of central atoms

        atom_type_a : integer (or char)
            type of interacting atoms

        rcut_e : float
            external of the shell

        rcut_i : float
            internal of the shell
        """
        self.natoms = natoms
        self.box_size = box_size
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b
        self.rcut_e = rcut_e
        self.rcut_i = rcut_i

        N_x = np.count_nonzero(atom_type == self.atom_type_a)
        self.cn = np.zeros(N_x)
        self.ncn = 0
        self.bound = boundary.apply(self.box_size)
        

    def accumulate(self, atom_type, positions):
        """
        Parameters
        ----------
        atom_type : numpy array with integers (could be char)
            type of atoms
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z
        """

        idx = 0
        for i in range(0, self.natoms):

            if (atom_type[i] != self.atom_type_a): continue
            ri = [positions[i + k*self.natoms] for k in range(0,3)]
            
            for j in range(0, self.natoms):

                if (j == i): continue
                
                if (atom_type[j] != self.atom_type_b): continue
                rj = [positions[j + k*self.natoms] for k in range(0,3)]

                rij = self.bound.minimum_image(ri, rj)
                rij = np.linalg.norm(rij)

                if (rij <= self.rcut_i or rij > self.rcut_e): continue
                self.cn[idx] += 1
                    
            idx += 1

        self.ncn +=1
    

    def end(self, atom_type, positions, writes=True, file_cn='cn.dat'):
        """
        Parameters
        ----------
        writes : True (or False)
            if you want (or don't want) to write an output

        file_cn : filname
            the file were the cn is going to be written
        
        positions : numpy array with float32 data
            the positions in the SoA convention
            i.e. first all the x, then y and then z

        Returns
        -------
        self.cn : numpy array
            an array with the coordination number of each atom selected
        """

        if (writes == True):
            file_cn = open(file_cn, 'w')
            file_cn.write("%d\n\n" % len(self.cn))

            idx = 0
            for i in range(0, self.natoms):
            
                if (atom_type[i] != self.atom_type_a): continue
                
                file_cn.write("%d %g %g %g %g\n" % (atom_type[i], positions[i], \
                    positions[self.natoms + i], positions[2*self.natoms + i], \
                    self.cn[idx]/self.ncn))
                
                idx += 1

            file_cn.close()

        return np.array(self.cn / self.ncn, dtype=np.float32)
