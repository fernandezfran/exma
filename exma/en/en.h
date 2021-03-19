#ifndef EN_H
#define EN_H

#include <math.h>

void distance_matrix(const int N_central, const int N_interact,
                     const float *box_size, const float *x_central,
                     const float *x_interact, float *distrix);

#endif
