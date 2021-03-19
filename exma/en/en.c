#include "en.h"

void distance_matrix(const int N_central, const int N_interact,
                     const float *box_size, const float *x_central,
                     const float *x_interact, float *distrix){
    /* calculate the distance matrix using minimum images */
    float ri[3], rj[3];
    float r2, rij;

    for (int k = 0; k < (N_central * N_interact); k++) distrix[k] = 0.0f;

    for (int i = 0; i < N_interact; i++){

        for (int k = 0; k < 3; k++) ri[k] = x_interact[k*N_interact + i];

        for (int j = 0; j < N_central ; j++){
        
            r2 = 0.0f;
            for (int k = 0; k < 3; k++){
                rj[k] = x_central[k*N_central + j];
                rij = rj[k] - ri[k];
                while (rij > 0.5f * box_size[k]) rij -= box_size[k];
                while (rij < -0.5f * box_size[k]) rij += box_size[k];
                r2 += rij * rij;
            }
            rij = sqrt(r2);
            
            distrix[N_central * i + j] = rij;
        }
    }
}
