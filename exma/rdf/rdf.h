#ifndef RDF_H
#define RDF_H

#include <math.h>

void monoatomic(const int N, const float *box_size, const float *positions, 
                const float dg, const int nbin, int *gr);

void diatomic(const int N, const float *box_size, const int *atom_type,
              const int atype_a, const int atype_b, const float *positions, 
              const float dg, const int nbin, int *gr);

#endif
