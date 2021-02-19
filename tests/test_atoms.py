import numpy as np

from exma import atoms

particles = atoms.positions(4, 1.0)
print(np.transpose(np.split(particles.fcc(),3)))
