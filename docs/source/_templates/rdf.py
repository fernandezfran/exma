#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Python script to calculate the RDF of a LJ fluid in a solid and in a liquid 
#   phase
#
import numpy as np
import matplotlib.pyplot as plt

import exma


# solid phase
N = 500

frames = 201
ssize = np.full(3, 7.46901) 

solid = exma.reader.xyz("../_static/lj-fcc.xyz")
srdf = exma.RDF.rdf.monoatomic(N, ssize, 75)

for i in range(0, frames):
    sN, styp, sx = solid.read_frame()
    srdf.accumulate(sx)

sr, sgofr = srdf.end(False)
solid.file_close()


# solid phase
frames = 201
lsize = np.full(3, 8.54988) 

liquid = exma.reader.xyz("../_static/lj-liquid.xyz")
lrdf = exma.RDF.rdf.monoatomic(N, lsize, 75)

for i in range(0, frames):
    lN, ltyp, lx = liquid.read_frame()
    lrdf.accumulate(lx)

lr, lgofr = lrdf.end(False)
liquid.file_close()


# graphic
plt.xlabel("r*")
plt.ylabel("g(r)")
plt.xlim(0.0, 4.0)
plt.hlines(1.0, 0.0, 4.0, colors='k', ls='dashed')
plt.plot(sr, sgofr, label='solid')
plt.plot(lr, lgofr, label='liquid')
plt.legend()
plt.savefig('rdf.png', dpi=600)
plt.show()
