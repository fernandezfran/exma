#ifndef RDF_H
#define RDF_H

#include <math.h>

void rdf_acumulate(const int natoms_c, const int natoms_i, const float *box,
                   const float *x_central, const float *x_interact,
                   const int pbc, const float dg, const float rmax,
                   const int nbin, int *gr);

#endif
