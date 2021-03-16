#include "rdf.h"

void monoatomic(const int N, const float *box_size, const float *positions, 
                const float dg, const int nbin, int *gr) {
    /* Calculate the rdf of a monoatomic system for the actual frame */
    float ri[3], rj[3];
    float rij, rij2;
    float rmax = (float) nbin * dg;
    int ig;

    for (int i = 0; i < N; i++) {

        for (int k = 0; k < 3; k++) ri[k] = positions[k*N + i]; 

        for (int j = i+1; j < N; j++) {
            
            rij2 = 0.0f;
            for (int k = 0; k < 3; k++) {
                rj[k] = positions[k*N + j];
                rij = rj[k] - ri[k];
                while (rij > 0.5f * box_size[k]) rij -= box_size[k];
                while (rij < -0.5f * box_size[k]) rij += box_size[k];
                rij2 += rij * rij;
            }
            rij = sqrt(rij2);

            if (rij > rmax) continue;

            ig = (int) (rij / dg);
            gr[ig] += 2;
        }
    }
}
