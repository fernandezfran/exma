// This file is part of exma (https://github.com/fernandezfran/exma/)
// Copyright (c) 2021, Francisco Fernandez
// License: MIT
//     Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE
#ifndef CN_H
#define CN_H

void cn_accumulate(const int natoms_c, const int natoms_i, const float *box,
                   const float *x_central, const float *x_interact,
                   const int pbc, const float rcut_i, const float rcut_e,
                   int *cn);

#endif
