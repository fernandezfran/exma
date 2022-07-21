// This file is part of exma (https://github.com/fernandezfran/exma/)
// Copyright (c) 2021, Francisco Fernandez
// License: MIT
//     Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE
#ifndef RDF_H
#define RDF_H

void rdf_acumulate(const int natoms_c, const int natoms_i, const float *box,
                   const float *x_central, const float *x_interact,
                   const int pbc, const float dg, const float rmax,
                   const int nbin, int *gr);

#endif
