#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Python script to calculate the MSD of a LJ fluid in a solid and in a liquid 
#   phase
#
import time
import numpy as np
import matplotlib.pyplot as plt

import exma

# solid phase
N = 500

frames = 201
ssize = np.full(3, 7.46901) 

start_time = time.time()

solid = exma.reader.xyz("../_static/lj-fcc.xyz", 'image')

sN, styp, sx, simg = solid.read_frame()
sMSD = exma.msd.monoatomic(N, sx)

st, smsd = [], []
for i in range(0, frames - 1):
    sN, styp, sx, simg = solid.read_frame()
    t, msd = sMSD.wrapped(ssize, sx, simg)

    st.append(t)
    smsd.append(msd)

solid.file_close()
st = np.asarray(st)
smsd = np.asarray(smsd)


# solid phase
frames = 201
lsize = np.full(3, 8.54988) 

liquid = exma.reader.xyz("../_static/lj-liquid.xyz", 'image')

lN, ltyp, lx, limg = liquid.read_frame()
lMSD = exma.msd.monoatomic(N, lx)

lt, lmsd = [], []
for i in range(0, frames - 1):
    lN, ltyp, lx, limg = liquid.read_frame()
    t, msd = lMSD.wrapped(lsize, lx, limg)

    lt.append(t)
    lmsd.append(msd)

liquid.file_close()
lt = np.asarray(lt)
lmsd = np.asarray(lmsd)

print("--- %s sec ---" % (time.time() - start_time))

# graphic
plt.xlabel("frames")
plt.ylabel("MSD")
plt.plot(st, smsd, label='solid')
plt.plot(lt, lmsd, label='liquid')
plt.legend()
plt.savefig('msd.png', dpi=600)
plt.show()
