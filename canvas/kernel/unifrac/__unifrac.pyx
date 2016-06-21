# encoding: utf-8
# cython: profile=True
# filename: unifrac.pyx

import numpy as np
cimport cython
cimport numpy as np

@cython.boundscheck(False)
# N, D = 20, 20
# 1000 loops, best of 3: 206 µs per loop
def _fast_pairwise_weighted_logaitchison(double [:, :] X,
                                         double [:, :] tdm,
                                         double [:, :] out):
    cdef int N, D
    N = X.shape[0]
    D = X.shape[1]
    for i in range(N):
        for j in range(i):
            for u in range(D):
                for v in range(u):
                    out[i, j] += tdm[u, v] * \
                           (((X[i, u] - X[i, v]) - (X[j, u] - X[j, v])) * \
                            ((X[i, u] - X[i, v]) - (X[j, u] - X[j, v])))
            out[i, j] = np.sqrt(out[i, j] / D)
