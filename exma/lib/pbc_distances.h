// This file is part of exma (https://github.com/fernandezfran/exma/)
// Copyright (c) 2021, Francisco Fernandez
// License: MIT
//     Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE
#ifndef PBC_DISTANCES_H
#define PBC_DISTANCES_H

void distance_matrix(const int natoms_c, const int natoms_i, const float *box,
                     const float *x_central, const float *x_interact,
                     float *distrix);

#endif
