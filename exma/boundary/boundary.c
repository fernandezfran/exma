#include "boundary.h"
#include <stdio.h>

void pbc(const int N, const float *box_size, float *x){
    /* periodic boundary conditions */
    for (int j = 0; j < 3; j++){
        for (int i = 0; i < N; i++){
            while (x[j*N + i] < 0.0) x[j*N + i] += box_size[j];
            while (x[j*N + i] > box_size[j]) x[j*N + i] -= box_size[j];
        }
    }
}


void minimum_image(const float *box_size, const float *central,
                   const float *interact, float *x_ci){
    /* minimum image */
    for (int i = 0; i < 3; i++){
        x_ci[i] = interact[i] - central[i];
        while (x_ci[i] > 0.5f * box_size[i]) x_ci[i] -= box_size[i];
        while (x_ci[i] < -0.5f * box_size[i]) x_ci[i] += box_size[i];
    }
}
