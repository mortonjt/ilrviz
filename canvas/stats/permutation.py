from __future__ import division
import numpy as np
import pandas as pd
from time import time
import copy
from scipy.stats import ttest_ind, f_oneway
from scipy._lib._util import check_random_state
from canvas.util import check_table_grouping
import itertools


def _init_perms(vec, permutations=1000, random_state=None):
    """
    Creates a permutation matrix

    vec: numpy.array
       Array of values to be permuted
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    random_state = check_random_state(random_state)
    c = len(vec)
    copy_vec = copy.deepcopy(vec)
    perms = np.array(np.zeros((c, permutations+1), dtype=np.float64))
    _samp_ones = np.array(np.ones(c), dtype=np.float64).transpose()
    for m in range(permutations+1):
        perms[:,m] = copy_vec
        random_state.shuffle(copy_vec)
    return perms

def _init_categorical_perms(cats, permutations=1000, random_state=None):
    """
    Creates a reciprocal permutation matrix

    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    random_state = check_random_state(random_state)
    c = len(cats)
    num_cats = len(np.unique(cats)) # Number of distinct categories
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)), dtype=np.float64))
    for m in range(permutations+1):
        for i in range(num_cats):
            perms[:,num_cats*m+i] = (copy_cats == i).astype(np.float64)
        random_state.shuffle(copy_cats)
    return perms

def _init_reciprocal_perms(cats, permutations=1000, random_state=None):
    """
    Creates a reciprocal permutation matrix.
    This is to ease the process of division.

    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    random_state = check_random_state(random_state)
    num_cats = len(np.unique(cats)) #number of distinct categories
    c = len(cats)
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)), dtype=np.float64))
    _samp_ones = np.array(np.ones(c), dtype=np.float64).transpose()
    for m in range(permutations+1):

        #Perform division to make mean calculation easier
        perms[:,2*m] = copy_cats / float(copy_cats.sum())
        perms[:,2*m+1] = (_samp_ones - copy_cats) / float((_samp_ones - copy_cats).sum())
        random_state.shuffle(copy_cats)

    return perms


############################################################
## Mean permutation tests
############################################################

def _naive_mean_permutation_test(mat,cats,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now

    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    the naive approach
    """
    def _mean_test(values,cats):
        #calculates mean for binary categories
        return abs(values[cats==0].mean()-values[cats==1].mean())

    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r,:].transpose()
        test_stat = _mean_test(values,cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _mean_test(values,perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues

def fisher_mean(table, grouping, permutations=1000, random_state=None):
    """ Conducts a fishers test on a contingency table.

    This module will conduct a mean permutation test using
    numpy matrix algebra.

    table: pd.DataFrame
        Contingency table of where columns correspond to features
        and rows correspond to samples.
    grouping : pd.Series
        Vector indicating the assignment of samples to groups.  For example,
        these could be strings or integers denoting which group a sample
        belongs to.  It must be the same length as the samples in `table`.
        The index must be the same on `table` and `grouping` but need not be
        in the same order.
    permutations: int
         Number of permutations to calculate
    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.

    Return
    ------
    pd.DataFrame
        A table of features, their t-statistics and p-values
        `"m"` is the t-statistic.
        `"pvalue"` is the p-value calculated from the permutation test.

    Examples
    --------
    >>> from canvas.stats.permutation import fisher_mean_test
    >>> import pandas as pd
    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [22, 21, 9,  10, 10, 10, 10],
    ...                       [20, 22, 10, 10, 13, 10, 10],
    ...                       [23, 21, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])
    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])
    >>> results = fisher_mean_test(table, grouping,
    ...                            permutations=100, random_state=0)
    >>> results
                m    pvalue
    b1  14.333333  0.108910
    b2  10.333333  0.108910
    b3   0.333333  1.000000
    b4   0.333333  1.000000
    b5   1.000000  1.000000
    b6   1.666667  0.108910
    b7   0.333333  1.000000

    Notes
    -----
    Only works on binary classes.
    """

    mat, cats = check_table_grouping(table, grouping)
    perms = _init_reciprocal_perms(cats.values, permutations,
                                   random_state=random_state)

    m, p = _np_two_sample_mean_statistic(mat.values.T, perms)
    res = pd.DataFrame({'m': m, 'pvalue': p}, index=mat.columns)

    return res

def _np_two_sample_mean_statistic(mat, perms):
    """
    Calculates a permutative mean statistic just looking at binary classes

    mat: numpy.ndarray or scipy.sparse.*
         Contingency table. Eolumns correspond to features (e.g. OTUs)
         and rows correspond to samples.

    perms: numpy.ndarray
         Permutative matrix.
         Columns correspond to  permutations of samples
         rows corresponds to features

    Note: only works on binary classes now

    Returns
    =======
    m:
        List of mean test statistics
    p:
        List of corrected p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """

    ## Create a permutation matrix
    num_cats = 2 #number of distinct categories
    n_otus, c = perms.shape
    permutations = (c-num_cats) // num_cats

    ## Perform matrix multiplication on data matrix
    ## and calculate averages
    avgs = mat.dot(perms)
    ## Calculate the mean statistic
    idx = np.arange(0, (permutations+1)*num_cats, num_cats)
    mean_stat = abs(avgs[:, idx+1] - avgs[:, idx])

    ## Calculate the p-values
    cmps =  (mean_stat[:,1:].T >= mean_stat[:,0]).T
    pvalues = (cmps.sum(axis=1)+1.)/(permutations+1.)

    #_,pvalues,_,_ = multipletests(pvalues)
    m = np.ravel(mean_stat[:,0])
    p = np.array(np.ravel(pvalues))
    return m, p



############################################################
## T-test permutation tests
############################################################
def _naive_t_permutation_test(mat,cats,permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now

    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    the naive approach
    """
    def _t_test(values,cats):
        #calculates t statistic for binary categories
        T, _ =  ttest_ind(values[cats==0], values[cats==1], equal_var = False)
        return abs(T)

    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r,:].transpose()
        test_stat = _t_test(values,cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _t_test(values,perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    #_,pvalues,_,_ = multipletests(pvalues)
    return test_stats, pvalues


def permutative_ttest(table, grouping,
                      permutations=1000,
                      equal_var=False,
                      random_state=None):
    """ Performs permutative ttest

    This module will conduct a mean permutation test using
    numpy matrix algebra.
    table : pd.DataFrame
        A 2D matrix of strictly positive values (i.e. counts or proportions)
        where the rows correspond to samples and the columns correspond to
        features.
    grouping : pd.Series
        Vector indicating the assignment of samples to groups.  For example,
        these could be strings or integers denoting which group a sample
        belongs to.  It must be the same length as the samples in `table`.
        The index must be the same on `table` and `grouping` but need not be
        in the same order.
    permutations: int
         Number of permutations to calculate.
    equal_var: bool, optional
        If false, a Welch's t-test is conducted.  Otherwise, an ordinary t-test
        is conducted.
    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.

    Return
    ------
    pd.DataFrame
        A table of features, their t-statistics and p-values
        `"t"` is the t-statistic.
        `"pvalue"` is the p-value calculated from the permutation test.

    Examples
    --------
    >>> import pandas as pd
    >>> from canvas.stats.permutation import permutative_ttest
    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [22, 21, 9,  10, 10, 10, 10],
    ...                       [20, 22, 10, 10, 13, 10, 10],
    ...                       [23, 21, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])
    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])
    >>> results = permutative_ttest(table, grouping,
    ...                       permutations=100, random_state=0)
    >>> results
          pvalue          t
    b1  0.108911   4.216497
    b2  0.108911  31.000000
    b3  1.000000   0.200000
    b4  1.000000   1.000000
    b5  1.000000   1.000000
    b6  1.000000   1.000000
    b7  1.000000   1.000000
    """
    mat, cats = check_table_grouping(table, grouping)
    perms = _init_categorical_perms(cats, permutations, random_state)
    t, p = _np_two_sample_t_statistic(mat.T.values, perms)
    res = pd.DataFrame({'t': t, 'pvalue': p}, index=table.columns)
    return res

def _np_two_sample_t_statistic(mat, perms, equal_var=False):
    """
    Calculates a permutative Welch's t-statistic

    mat: numpy.matrix or scipy.sparse.*
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    perms: numpy.matrix
         columns: permutations of samples
         rows: features
         Permutative matrix
    equal_var: bool, optional
        If false, a Welch's t-test is conducted.  Otherwise, an ordinary t-test
        is conducted.
    Note: only works on binary classes now

    Returns
    =======
    t:
        List of t-test statistics
    p:
        List of p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra.
    """

    ## Create a permutation matrix
    num_cats = 2 # number of distinct categories
    n_otus, c = perms.shape
    permutations = (c-num_cats) // num_cats

    ## Perform matrix multiplication on data matrix
    ## and calculate sums and squared sums
    _sums  = mat.dot(perms)
    _sums2 = np.multiply(mat,mat).dot(perms)

    ## Calculate means and sample variances
    tot =  perms.sum(axis=0)
    _avgs  = _sums / tot
    _avgs2 = _sums2 / tot
    _vars  = _avgs2 - np.multiply(_avgs, _avgs)
    _samp_vars =  np.multiply(tot,_vars) / (tot-1)

    ## Calculate the t statistic
    idx = np.arange(0, (permutations+1)*num_cats, num_cats)
    # denom  = np.sqrt(_samp_vars[:, idx+1] / tot[:,idx+1]  + _samp_vars[:, idx] / tot[:,idx])

    # Calculate the t statistic
    if not equal_var:
        denom = np.sqrt(np.divide(_samp_vars[:, idx+1], tot[idx+1]) +
                        np.divide(_samp_vars[:, idx], tot[idx]))
    else:
        df = tot[idx] + tot[idx+1] - 2
        svar = ((tot[idx+1] - 1) * _samp_vars[:, idx+1] + (tot[idx] - 1) *
                _samp_vars[:, idx]) / df
        denom = np.sqrt(svar * (1.0 / tot[idx+1] + 1.0 / tot[idx]))

    t_stat = np.divide(abs(_avgs[:, idx+1] - _avgs[:, idx]), denom)

    ## Calculate the p-values
    cmps =  t_stat[:,1:].T >= t_stat[:,0]
    pvalues = (cmps.sum(axis=0)+1.)/(permutations+1.)
    t = np.ravel(t_stat[:,0])
    p = np.array(pvalues)
    return t, p


############################################################
## F-test permutation tests
############################################################

def _naive_f_permutation_test(mat,cats,permutations=1000):
    """
    Performs a 1-way ANOVA.

    F = sum( MS_i for all i) /  MSE

    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a F permutation test using
    the naive approach
    """

    def _f_test(values,cats):
        #calculates t statistic for binary categories
        groups = []
        groups = [ values[cats==k] for k in set(cats) ]
        F, _ =  f_oneway(*groups)
        return abs(F)

    rows,cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r,:].transpose()
        test_stat = _f_test(values,cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _f_test(values,perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    return test_stats, pvalues

def permutative_anova(table, grouping, permutations=1000, random_state=None):
    """
    Calculates a permutative one way anova.
    table : pd.DataFrame
        A 2D matrix of strictly positive values (i.e. counts or proportions)
        where the rows correspond to samples and the columns correspond to
        features.
    grouping : pd.Series
        Vector indicating the assignment of samples to groups.  For example,
        these could be strings or integers denoting which group a sample
        belongs to.  It must be the same length as the samples in `table`.
        The index must be the same on `table` and `grouping` but need not be
        in the same order.
    permutations: int
       Number of permutations for permutation test
    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.

    Returns
    =======
    pd.DataFrame
        A table of features, their t-statistics and p-values
        `"f"` is the f-statistic.
        `"pvalue"` is the p-value calculated from the permutation test.

    Examples
    --------
    >>> import pandas as pd
    >>> from canvas.stats.permutation import permutative_anova
    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [22, 21, 9,  10, 10, 10, 10],
    ...                       [20, 22, 10, 10, 13, 10, 10],
    ...                       [23, 21, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])
    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])
    >>> results = permutative_anova(table, grouping,
    ...                       permutations=100, random_state=0)
    >>> results
                 f    pvalue
    b1   17.778846  0.108911
    b2  961.000000  0.108911
    b3    0.040000  1.000000
    b4    1.000000  1.000000
    b5    1.000000  1.000000
    b6    1.000000  1.000000
    b7    1.000000  1.000000

    This module will conduct a mean permutation test using
    numpy matrix algebra.
    """
    mat, cats = check_table_grouping(table, grouping)
    perms = _init_categorical_perms(cats, permutations, random_state)
    f, p = _np_k_sample_f_statistic(mat.values.T, cats, perms)
    res = pd.DataFrame({'f': f, 'pvalue': p}, index=table.columns)
    return res


def _np_k_sample_f_statistic(mat, cats, perms):
    """
    Calculates a permutative one way F test

    mat: numpy.array
         The contingency table.
         Columns correspond to features (e.g. OTUs)
         and rows correspond to  samples
    cat : numpy.array
         Vector of categories.
    perms: numpy.array
         Permutative matrix. Columns correspond to permutations
         of samples rows corespond to features

    Returns
    =======
    test_stats:
        List of f-test statistics
    pvalues:
        List of p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """

    ## Create a permutation matrix
    num_cats = len(np.unique(cats)) # Number of distinct categories
    n_samp, c = perms.shape
    permutations = (c-num_cats) / num_cats

    mat2 = np.multiply(mat, mat)

    S = mat.sum(axis=1)
    SS = mat2.sum(axis=1)
    sstot = SS - np.multiply(S,S) / float(n_samp)
    #Create index to sum the ssE together
    _sum_idx = _init_categorical_perms(
        np.arange((permutations+1)*num_cats,dtype=np.int32) // num_cats,
        permutations=0)


    ## Perform matrix multiplication on data matrix
    ## and calculate sums and squared sums and sum of squares
    _sums  = np.dot(mat, perms)
    _sums2 = np.dot(np.multiply(mat, mat), perms)
    tot =  perms.sum(axis=0)
    ss = _sums2 - np.multiply(_sums,_sums)/tot
    sserr = np.dot(ss, _sum_idx)
    sstrt = (sstot - sserr.T).T

    dftrt = num_cats-1
    dferr = np.dot(tot,_sum_idx) - num_cats
    f_stat = (sstrt / dftrt) / (sserr / dferr)

    cmps =  f_stat[:,1:].T >= f_stat[:,0]
    pvalues = (cmps.sum(axis=0)+1.) / (permutations+1.)
    f = np.array(np.ravel(f_stat[:, 0]))
    p = np.array(np.ravel(pvalues))
    return f, p

