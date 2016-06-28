from __future__ import division
from canvas.phylogeny import phylogenetic_basis
from skbio.stats.composition import ilr
from ete3 import Tree, TreeStyle, faces, AttrFace, CircleFace, BarChartFace
import numpy as np
import pandas as pd
import scipy


def default_layout(node):
    if node.is_leaf():
        # Add node name to leaf nodes
        N = AttrFace("name", fsize=14, fgcolor="black")

        faces.add_face_to_node(N, node, 0)
    if "weight" in node.features:
        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        C = CircleFace(radius=node.weight, color="Red", style="sphere")
        # Let's make the sphere transparent
        C.opacity = 0.5
        # Rotate the faces by 90*
        C.rotation = 90
        # And place as a float face over the tree
        faces.add_face_to_node(C, node, 0, position="float")


def barchart_layout(node, name='name',
                    width=20, height=40,
                    colors=None, min_value=0, max_value=1,
                    fsize=14, fgcolor="black"):
    if colors is None:
        colors = ['#0000FF']
    if node.is_leaf():
        # Add node name to leaf nodes
        N = AttrFace("name", fsize=fsize, fgcolor=fgcolor)

        faces.add_face_to_node(N, node, 0)
    if "weight" in node.features:
        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        if (isinstance(node.weight, int) or isinstance(node.weight, float)):
            weight = [node.weight]
        else:
            weight = node.weight
        C = BarChartFace(values=weight, width=width, height=height,
                         colors=['#0000FF'], min_value=min_value,
                         max_value=max_value)
        # Let's make the sphere transparent
        C.opacity = 0.5
        # Rotate the faces by 270*
        C.rotation = 270
        # And place as a float face over the tree
        faces.add_face_to_node(C, node, 0, position="float")


def balanceplot(balances, tree,
                layout=None,
                mode='c'):
    """ Plots balances on tree.

    Parameters
    ----------
    balances : np.array
        A vector of internal nodes and their associated real-valued balances.
        The order of the balances will be assumed to be in level order.
    tree : skbio.TreeNode
        A strictly bifurcating tree defining a hierarchical relationship
        between all of the features within `table`.
    layout : function, optional
        A layout for formatting the tree visualization. Must take a
        `ete.tree` as a parameter.
    mode : str
        Type of display to show the tree. ('c': circular, 'r': rectangular).

    Note
    ----
    The `tree` is assumed to strictly bifurcating and
    whose tips match `balances.

    See Also
    --------
    TreeNode.levelorder
    """
    # The names aren't preserved - let's pray that the topology is consistent.
    ete_tree = Tree(str(tree))
    # Some random features in all nodes
    i = 0
    for n in ete_tree.traverse():
        if not n.is_leaf():
            n.add_features(weight=balances[-i])
            i += 1

    # Create an empty TreeStyle
    ts = TreeStyle()

    # Set our custom layout function
    if layout is None:
        ts.layout_fn = default_layout
    else:
        ts.layout_fn = layout
    # Draw a tree
    ts.mode = mode

    # We will add node names manually
    ts.show_leaf_name = False
    # Show branch data
    ts.show_branch_length = True
    ts.show_branch_support = True

    return ete_tree, ts


def balancetest(table, grouping, tree,
                significance_test=None,
                layout=None,
                normalize=True,
                mode='c'):
    """ Performs statistical test on ilr balances and plots on tree.

    Parameters
    ----------
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
    tree : skbio.TreeNode
        A strictly bifurcating tree defining a hierarchical relationship
        between all of the features within `table`
    significance_test : function, optional
        A statistical significance function to test for significance between
        classes.  This function must be able to accept at least two 1D
        array_like arguments of floats and returns a test statistic and a
        p-value, or a single statistic. By default ``scipy.stats.f_oneway``
        is used.
    layout : function, optional
        A layout for formatting the tree visualization. Must take a
        `ete.tree` as a parameter.
    mode : str
        Type of display to show the tree. ('c': circular, 'r': rectangular).

    Returns
    -------
    ete_tree : ete.Tree
        ETE tree converted from the `skbio.TreeNode` object
    ts : ete.TreeStyle
        ETE tree style used for formatting the visualized tree,
        with the test statistic plotted on each of the internal nodes.

    Note
    ----
    The `skbio.TreeNode` is assumed to strictly bifurcating and
    whose tips match `table`.  Also, it is assumed that none
    of the values in `table` are zero.  Replace with a pseudocount
    if necessary.

    See also
    --------
    skbio.TreeNode.bifurcate
    skbio.stats.composition.ilr
    skbio.stats.multiplicative_replacement
    scipy.stats.f_oneway
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError('`table` must be a `pd.DataFrame`, '
                        'not %r.' % type(table).__name__)
    if not isinstance(grouping, pd.Series):
        raise TypeError('`grouping` must be a `pd.Series`,'
                        ' not %r.' % type(grouping).__name__)

    if np.any(table <= 0):
        raise ValueError('Cannot handle zeros or negative values in `table`. '
                         'Use pseudo counts or ``multiplicative_replacement``.'
                         )

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

    if significance_test is None:
        significance_test = scipy.stats.f_oneway

    sorted_features = [n.name for n in tree.tips()][::-1]
    if len(sorted_features) != len(table.columns):
        raise ValueError('The number of tips (%d) in the tree must be equal '
                         'to the number features in the table (%d).' %
                         (len(sorted_features), len(table.columns)))
    table = table.reindex(columns=sorted_features)
    table_index_len = len(table.index)
    grouping_index_len = len(grouping.index)
    mat, cats = table.align(grouping, axis=0, join='inner')
    if (len(mat) != table_index_len or len(cats) != grouping_index_len):
        raise ValueError('`table` index and `grouping` '
                         'index must be consistent.')

    basis, nodes = phylogenetic_basis(tree)
    ilr_coords = ilr(mat, basis=basis)

    ete_tree = Tree(str(tree))

    _cats = set(grouping)
    i = 0
    for n in ete_tree.traverse():
        if not n.is_leaf():
            diffs = [ilr_coords[(cats == x).values, i] for x in _cats]

            stat = significance_test(*diffs)
            if len(stat) == 2:
                n.add_features(weight=-np.log(stat[1]))
            elif len(stat) == 1:
                n.add_features(weight=stat)
            else:
                raise ValueError(
                    "Too many arguments returned by %s" %
                    significance_test.__name__)
            i += 1

    # Create an empty TreeStyle
    ts = TreeStyle()

    # Set our custom layout function
    if layout is None:
        ts.layout_fn = default_layout
    else:
        ts.layout_fn = layout

    # Draw a tree
    ts.mode = mode

    # We will add node names manually
    ts.show_leaf_name = False
    # Show branch data
    ts.show_branch_length = True
    ts.show_branch_support = True

    return ete_tree, ts
