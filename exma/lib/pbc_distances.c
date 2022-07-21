// This file is part of exma (https://github.com/fernandezfran/exma/)
// Copyright (c) 2021, Francisco Fernandez
// License: MIT
//     Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE
#include "pbc_distances.h"

#include <math.h>

void distance_matrix(const int natoms_c, const int natoms_i, const float *box,
                     const float *x_central, const float *x_interact,
                     float *distrix) {
    /* calculate the distance matrix considering pbc.
     *
     * natoms_c : number of central atoms
     * natoms_i : number of interact atoms
     * box : array with the box lenght in each x, y, z direction
     * x_central : positions of the central atoms arranged as SoA (i.e. first
     *     all the x, then all y, and finally all z coordinates)
     * x_interact : positions of the interacting atoms arranged as SoA
     * distrix : array with a vector where the first natoms_i components
     *     are the distances of the interacting atoms to the first central one,
     *     then the second natoms_i to the second central atom, etc.
     */
    float ri[3], rj[3];
    float rij, rij2;

    // i'm standing on the central atoms
    for (int i = 0; i < natoms_c; ++i) {

        // select the vector position of a particular one
        for (int k = 0; k < 3; ++k) {
            ri[k] = x_central[k * natoms_c + i];
        }

        // computes the distance to all interacting atoms
        for (int j = 0; j < natoms_i; ++j) {
            rij2 = 0.0f;
            for (int k = 0; k < 3; ++k) {
                rj[k] = x_interact[k * natoms_i + j];
                rij = rj[k] - ri[k];
                while (rij > 0.5f * box[k])
                    rij -= box[k];
                while (rij < -0.5f * box[k])
                    rij += box[k];
                rij2 += rij * rij;
            }

            // save the distance in the matrix
            distrix[natoms_i * i + j] = sqrt(rij2);
        }
    }
}
