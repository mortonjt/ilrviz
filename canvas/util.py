import pandas as pd
import numpy as np


def check_table_grouping(table, grouping):
    if not isinstance(table, pd.DataFrame):
        raise TypeError('`table` must be a `pd.DataFrame`, '
                        'not %r.' % type(table).__name__)
    if not isinstance(grouping, pd.Series):
        raise TypeError('`grouping` must be a `pd.Series`,'
                        ' not %r.' % type(grouping).__name__)
    groups, _grouping = np.unique(grouping, return_inverse=True)
    grouping = pd.Series(_grouping, index=grouping.index)
    num_groups = len(groups)
    if num_groups == len(grouping):
        raise ValueError(
            "All values in `grouping` are unique. This method cannot "
            "operate on a grouping vector with only unique values (e.g., "
            "there are no 'within' variance because each group of samples "
            "contains only a single sample).")
    if num_groups == 1:
        raise ValueError(
            "All values the `grouping` are the same. This method cannot "
            "operate on a grouping vector with only a single group of samples"
            "(e.g., there are no 'between' variance because there is only a "
            "single group).")
    table_index_len = len(table.index)
    grouping_index_len = len(grouping.index)
    mat, cats = table.align(grouping, axis=0, join='inner')

    if (len(mat) != table_index_len or len(cats) != grouping_index_len):
        raise ValueError('`table` index and `grouping` '
                         'index must be consistent.')
    return mat, cats
