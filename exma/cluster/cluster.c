#include "cluster.h"

void distance_matrix(const int N, const float *box_size, const float *x,
                     float *distrix){
    /* calculate the distance matrix using minimum images */
    float ri[3], rj[3];
    float r2, rij;

    for (int k = 0; k < (N * N); k++) distrix[k] = 0.0f;

    for (int i = 0; i < N - 1; i++){

        for (int k = 0; k < 3; k++) ri[k] = x[k*N + i];

        for (int j = i + 1; j < N ; j++){
        
            r2 = 0.0f;
            for (int k = 0; k < 3; k++){
                rj[k] = x[k*N + j];
                rij = rj[k] - ri[k];
                while (rij > 0.5f * box_size[k]) rij -= box_size[k];
                while (rij < -0.5f * box_size[k]) rij += box_size[k];
                r2 += rij * rij;
            }
            rij = sqrt(r2);
            
            // the distance matrix is symetric
            distrix[N * i + j] = rij;
            distrix[N * j + i] = rij;
        }
    }
}
