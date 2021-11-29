#include "rdf.h"

void rdf_accumulate(const int natoms_c, const int natoms_i, const float *box,
                    const float *x_central, const float *x_interact,
                    const int pbc, const float dg, const float rmax,
                    const int nbin, int *gr) {
    /* calculate the rdf of central-interact atoms of a single frame.
     *
     * the data is accumulated in *gr, that must be initializated in zero
     * by the python main function.
     */
    float ri[3], rj[3];
    float rij2, rij;
    int ig;

    // i'm standing on the central atoms
    for (int i = 0; i < natoms_c; i++) {

        // select the vector position of a particular one
        for (int k = 0; k < 3; k++) {
            ri[k] = x_central[k * natoms_c + i];
        }

        // computes the distance to all interacting atoms
        for (int j = 0; j < natoms_i; j++) {
            rij2 = 0.0f;
            for (int k = 0; k < 3; k++) {
                rj[k] = x_interact[k * natoms_i + j];

                rij = rj[k] - ri[k];
                if (pbc == 1) {
                    while (rij > 0.5f * box[k])
                        rij -= box[k];
                    while (rij < -0.5f * box[k])
                        rij += box[k];
                }
                rij2 += rij * rij;
            }
            rij = sqrt(rij2);

            // accumulate in gr if the distance is less than the max
            if ((rij > 0) & (rij < rmax)) {
                ig = (int)(rij / dg);
                gr[ig] += 1;
            }
        }
    }
}
