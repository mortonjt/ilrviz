import pandas as pd
import numpy as np


def check_table_grouping(table, grouping):
    if not isinstance(table, pd.DataFrame):
        raise TypeError('`table` must be a `pd.DataFrame`, '
                        'not %r.' % type(table).__name__)
    if not isinstance(grouping, pd.Series):
        raise TypeError('`grouping` must be a `pd.Series`,'
                        ' not %r.' % type(grouping).__name__)

    if (grouping.isnull()).any():
        raise ValueError('Cannot handle missing values in `grouping`.')

    if (table.isnull()).any().any():
        raise ValueError('Cannot handle missing values in `table`.')

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
    print(len(mat), table_index_len)
    print(len(cats),  grouping_index_len)
    if (len(mat) != table_index_len or len(cats) != grouping_index_len):
        raise ValueError('`table` index and `grouping` '
                         'index must be consistent.')
    return mat, cats

def match(table, metadata):
    """ Returns common samples between metadata and table.

    Finds the samples that are common between the `table`,
    and the `metadata`, and returns the subset of the `table`
    and `metadata` with those samples.

    Parameters
    ----------
    table: pd.DataFrame
        Contingency table where columns correspond to features
        and rows correspond to samples.
    metadata : pd.DataFrame
        Metadata table where columns correspond to metadata
        and rows correspond to samples.

    Returns
    -------
    subtable: pd.DataFrame
        Contingency table where columns correspond to features
        and rows correspond to samples.  Contains only samples
        that are found in both subtable and submetadata.
    submetadata : pd.DataFrame
        Metadata table where columns correspond to metadata
        and rows correspond to samples.  Contains only samples
        that are found in both subtable and submetadata.

    """
    ids = list(set(table.index) & set(metadata.index))
    subtable = table.loc[ids]
    submetadata = metadata.loc[ids]
    return subtable, submetadata
