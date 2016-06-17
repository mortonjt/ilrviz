from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skbio.stats.composition import (ilr, ilr_inv,
                                     inner, perturb_inv)
from skbio import TreeNode
from scipy.spatial.distance import euclidean


def _rsq(predict, actual):
    """
    Calculates coefficient of determination

    Parameters
    ----------
    predict : numpy.ndarray, float
       a matrix of proportions where
       rows correspond to samples and
       columns correspond to features.
    """
    predict = np.atleast_2d(predict)
    actual = np.atleast_2d(actual)
    r, _ = predict.shape

    sst_hat = 0
    c =  actual.mean(axis=0)
    for i in range(r):
        sst_hat += euclidean(actual[i, :], c)
    ssr_hat = 0
    for i in range(r):
        ssr_hat += euclidean(predict[i, :], c)

    return ssr_hat / sst_hat


def simplicialOLS(Y, X, basis=None, formula=None):
    """
    Performs a simplicial ordinary least squares on a set of compositions
    and a response variable

    Parameters
    ----------
    y : numpy.ndarray, float
       a matrix of proportions where
       rows = compositions and
       columns = components
    X : numpy.ndarray, float
       independent variable

    Returns
    -------
    predict: pd.DataFrame, float
       a predicted matrix of proportions where
       rows correspond to samples and
       columns correspond to features.
    b: pd.DataFrame, float
       a matrix of estimated coefficient compositions
    resid: pd.DataFrame, float
       a matrix of residuals
    r2: float
       coefficient of determination

    See Also
    --------
    skbio.stats.composition.multiplicative_replacement

    Examples
    --------
    >>> import pandas as pd
    >>> from canvas.stats.regression import simplicialOLS
    >>> y = pd.DataFrame([[0.5       , 0.5       ],
                          [0.52631579, 0.47368421],
                          [0.55      , 0.45      ],
                          [0.57142857, 0.42857143],
                          [0.59090909, 0.40909091],
                          [0.60869565, 0.39130435],
                          [0.625     , 0.375     ],
                          [0.64      , 0.36      ],
                          [0.65384615, 0.34615385],
                          [0.66666667, 0.33333333]])
    >>> x = pd.DataFrame([[1. , 0.        ],
                          [1. , 1.11111111],
                          [1. , 2.22222222],
                          [1. , 3.33333333],
                          [1. , 4.44444444],
                          [1. , 5.55555556],
                          [1. , 6.66666667],
                          [1. , 7.77777778],
                          [1. , 8.88888889],
                          [1. , 10.]])
    >>> res = simplicialOLS(y, x)
    >>> res[1]
             b0        b1
    0  0.509761  0.490239
    1  0.517134  0.482866

    References
    ----------
    .. [1] Egozcue, J. J., et al. "Simplicial regression. The normal model."
    """
    if not isinstance(Y, pd.DataFrame):
        raise TypeError('`Y` must be a `pd.DataFrame`, '
                        'not %r.' % type(Y).__name__)
    if not isinstance(X, pd.DataFrame):
        raise TypeError('`X` must be a `pd.DataFrame`, '
                        'not %r.' % type(X).__name__)
    if np.any(Y <= 0):
        raise ValueError('Cannot handle zeros or negative values in `Y`. '
                         'Use pseudocounts or ``multiplicative_replacement``.'
                         )
    if (Y.isnull()).any().any():
        raise ValueError('Cannot handle missing values in `Y`.')

    if (X.isnull()).any().any():
        raise ValueError('Cannot handle missing values in `X`.')

    Y_index_len = len(Y.index)
    X_index_len = len(X.index)
    Y, X = Y.align(X, axis=0, join='inner')
    if (len(Y) != Y_index_len or len(X) != X_index_len):
        raise ValueError('`Y` index and `X` index must be consistent.')


    predict, b, err, r2 =  _regression(Y.values, X.values, basis=None)
    predict = pd.DataFrame(predict, index=Y.index, columns=Y.columns)

    var_ids = map(lambda x: 'b%d'%x, range(b.shape[-1]))

    b = pd.DataFrame(b, index=Y.columns, columns=var_ids)
    err = pd.DataFrame(err, index=Y.index, columns=Y.columns)
    return predict, b, err, r2


def _regression(y, X, basis=None):
    """
    Performs a simplicial ordinary least squares on a set of
    compositions and a response variable

    Parameters
    ----------
    y : numpy.ndarray, float
       a matrix of proportions where
       rows correspond to samples and
       columns correspond to features.
    X : numpy.ndarray, float
       independent variable

    Returns
    -------
    predict: pd.DataFrame, float
       a predicted matrix of proportions where
       rows correspond to samples and
       columns correspond to features.
    b: pd.DataFrame, float
       a matrix of estimated coefficient compositions
    resid: pd.DataFrame, float
       a matrix of compositional residuals
    r2: float
       coefficient of determination
    """
    y = np.atleast_2d(y)
    X = np.atleast_2d(X)

    # Need to add constant for intercept
    r, c = X.shape

    y_ = ilr(y, basis=basis)

    # Now perform least squares to calculate unknown coefficients
    inv = np.linalg.pinv(np.dot(X.T, X))
    cross = np.dot(inv, X.T)
    b_ = np.dot(cross, y_)
    predict_ = np.dot(X, b_)
    resid = (y_ - predict_)
    sst = (y_ - y_.mean(axis=0))
    r2 = 1 - ((resid**2).sum() / (sst**2).sum())

    if len(b_.shape) == 1:
        b_ = np.atleast_2d(b_).T

    b = ilr_inv(b_)
    if len(predict_.shape) == 1:
        predict_ = np.atleast_2d(predict_).T
    predict = ilr_inv(predict_)

    if len(resid.shape) == 1:
        resid = np.atleast_2d(resid).T
    resid = ilr_inv(resid)
    return predict, b, resid, r2
