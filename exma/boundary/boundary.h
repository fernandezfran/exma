#ifndef BOUNDARY_H
#define BOUNDARY_H

void pbc(const int N, const float *box_size, float *x);
void minimum_image(const float *box_size, const float *central,
                   const float *interact, float *x_ci);

#endif
