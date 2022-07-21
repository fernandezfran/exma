// This file is part of exma (https://github.com/fernandezfran/exma/)
// Copyright (c) 2021, Francisco Fernandez
// License: MIT
//     Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE
#include "rdf.h"

#include <math.h>

void rdf_accumulate(const int natoms_c, const int natoms_i, const float *box,
                    const float *x_central, const float *x_interact,
                    const int pbc, const float dg, const float rmax,
                    const int nbin, int *gr) {
    /* calculate the rdf of central-interact atoms of a single frame.
     *
     * natoms_c : number of central atoms
     * natoms_i : number of interact atoms
     * box : array with the box lenght in each x, y, z direction
     * x_central : positions of the central atoms arranged as SoA (i.e. first
     *     all the x, then all y, and finally all z coordinates)
     * x_interact : positions of the interacting atoms arranged as SoA
     * pbc : 1 if periodic boundary condition must be considered
     * dg : the width of each bin of the histogram
     * rmax : the maximum distance at which calculate the rdf
     * gr : array of ints were the data of the g(r) is accumulated, this must
     *     be initializated in python main function and normalized once it
     *     ends.
     */
    float ri[3], rj[3];
    float rij2, rij;
    int ig;

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
                if (pbc == 1) {
                    while (rij > 0.5f * box[k])
                        rij -= box[k];
                    while (rij < -0.5f * box[k])
                        rij += box[k];
                }
                rij2 += rij * rij;
            }
            rij = sqrt(rij2);

            // accumulate in gr if the distance is less than the max and
            // greater than zero (this is required if the central and interact
            // type of atoms are the same)
            if ((rij < rmax) & (rij > 0)) {
                ig = (int)(rij / dg);
                ++gr[ig];
            }
        }
    }
}
