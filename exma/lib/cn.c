// This file is part of exma (https://github.com/fernandezfran/exma/)
// Copyright (c) 2021, Francisco Fernandez
// License: MIT
//     Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE
#include "cn.h"

#include <math.h>

void cn_accumulate(const int natoms_c, const int natoms_i, const float *box,
                   const float *x_central, const float *x_interact,
                   const int pbc, const float rcut_i, const float rcut_e,
                   int *cn) {
    /* calculate the cn of central-interact atoms of a single frame.
     *
     * natoms_c : number of central atoms
     * natoms_i : number of interact atoms
     * box : array with the box lenght in each x, y, z direction
     * x_central : positions of the central atoms arranged as SoA (i.e. first
     *     all the x, then all y, and finally all z coordinates)
     * x_interact : positions of the interacting atoms arranged as SoA
     * pbc : 1 if periodic boundary condition must be considered
     * rcut_i : internal cut-off radius
     * rcut_e : external cut-off radius
     * cn : array of ints were the data of the CN is accumulated, this must
     *     be initializated in python main function and normalized once it
     *     ends.
     */
    float ri[3], rj[3];
    float rij2, rij;

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

            // accumulate in cn if the distance is less than the external
            // cut-off radius and greater than the internal cut-off radius.
            if ((rij <= rcut_e) & (rij > rcut_i)) ++cn[i];
        }
    }
}
