#ifndef CN_H
#define CN_H

#include <math.h>

void monoatomic(const int N, const float *box_size, const float *positions, 
                const float rcut_i, const float rcut_e, int *cn);

void diatomic(const int N, const float *box_size, const int *atom_type,
              const int atype_a, const int atype_b, const float *positions, 
              const float rcut_i, const float rcut_e, int *cn);

#endif
