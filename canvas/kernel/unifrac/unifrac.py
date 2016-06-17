from __future__ import division
import numpy as np
from skbio import DistanceMatrix
from copy import deepcopy
from canvas.kernel.unifrac._unifrac import _fast_pairwise_weighted_logaitchison


def aitchison_unifrac(x, y, dm):
    """ Uses a weighted Aitchison distance, weighted by
    tip-to-tip phylogenetic distance.

    Parameters
    ----------
    x : np.array
       Vector of observations from first sample
    y : np.array
       Vector of observations from second sample
    dm : np.array
       Distance matrix

    Returns
    -------
    float
       A distance between samples

    Notes
    -----
    The distance matrix needs to be in the same order as the vectors.
    Also, if there are any zeros, be sure to add a pseudocount.
    """
    _x, _y = np.log(x), np.log(y)
    D = len(x)
    d = 0
    for i in range(D):
        for j in range(i):
            d += dm[i, j] * ((_x[i] - _x[j]) - (_y[i] - _y[j]))**2
    return np.sqrt(d/D)


def _aitchison_unifrac_nolog(x, y, dm):
    """
    Parameters
    ----------
    x : np.array
       Vector of observations from first sample
    y : np.array
       Vector of observations from second sample
    dm : np.array
       Distance matrix

    Returns
    -------
    float
       A distance between samples

    Notes
    -----
    The distance matrix needs to be in the same order as the vectors.
    Also, if there are any zeros, be sure to add a pseudocount.
    """
    D = len(x)
    d = 0

    for i in range(D):
        for j in range(i):
            k = ((x[i] - x[j]) - (y[i] - y[j]))**2 * dm[i, j]
            d += k
    return np.sqrt(d/D)


def pairwise_aitchison_unifrac(X, tree):
    """ Calculates pairwise aitchison unifrac distance

    Parameters
    ----------
    X : pd.DataFrame
        Contingency table where row names correspond to samples and
        column names correspond to OTU ids.
    tree : skbio.TreeNode
        Phylogenetic tree

    Returns
    -------
    DistanceMatrix
        Distance matrix of weighted aitchison distances

    Examples
    --------
    >>> from skbio import TreeNode
    >>> import pandas as pd
    >>> from canvas.kernel.unifrac import pairwise_aitchison_unifrac
    >>> X = pd.DataFrame([[6, 3, 3], [4, 4, 4]],
    ...                  index=['sample1', 'sample2'],
    ...                  columns=['a', 'd', 'e'])
    >>> t = TreeNode.read([u"(a:1,(e:1, d:1)b:1)c;"])
    >>> res = pairwise_aitchison_unifrac(X, t)
    >>> res.data
    array([[ 0.        ,  0.98025814],
           [ 0.98025814,  0.        ]])
    """
    _X = deepcopy(X)
    sorted_otus = [n.name for n in tree.levelorder() if n.is_tip()]
    _X = _X.reindex_axis(sorted_otus, axis=1)

    N = X.shape[0]
    out = np.zeros((N, N))
    tree_dm = tree.tip_tip_distances()
    _fast_pairwise_weighted_logaitchison(np.log(X.values),
                                         tree_dm.data, out)
    dm = out
    return DistanceMatrix(dm+dm.T, ids=X.index)


def _pairwise_aitchison_unifrac_unrolled(X, dm, out):
    """ Mainly for testing and benchmarking purposes"""
    # N, D = 20, 20
    # 10 loops, best of 3: 38 ms per loop
    N = X.shape[0]
    D = X.shape[1]
    for i in range(N):
        for j in range(i):
            for u in range(D):
                for v in range(u):
                    out[i, j] += dm[u, v] * \
                           (((X[i, u] - X[i, v]) - (X[j, u] - X[j, v])) *
                            ((X[i, u] - X[i, v]) - (X[j, u] - X[j, v])))
            out[i, j] = np.sqrt(out[i, j] / D)
