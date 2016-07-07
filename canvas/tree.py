def _is_collapsible(tree):
    """ Checks to see if the children of the node can be collapsed. """
    if len(tree.children) == 0:
        return False

    for c in tree.children:
        if not c.is_tip():
            return False
    return True


def _trim_level(tree, table):
    """ Collapses tree leaves together by one level. """
    collapsed_tree = tree.copy()
    collapsed_table = table.copy()
    i = 0
    removed = set()
    for n in collapsed_tree.levelorder():
        if n in removed:
            continue
        if _is_collapsible(n):
            cs = n.children
            names = [c.name for c in cs]
            collapsed_feature = table[names].sum(axis=1)
            collapsed_table = collapsed_table.drop(names, axis=1)
            if n.name is not None:
                collapsed_table[n.name] = collapsed_feature
            else:
                collapsed_table[i] = collapsed_feature

            while len(n.children) > 0:
                c = n.pop()
                removed.add(c)
        i += 1
    return collapsed_tree, collapsed_table


def collapse(tree, table, level):
    """ Collapses tree leaves together.

    This functionality will collapse the tree leaves together.
    This will also collapse the corresponding features together
    in the contingency table.

    Parameters
    ----------
    tree : skbio.TreeNode
        Input tree.
    table : pd.DataFrame
        Contingency table where samples are rows and columns
        correspond to features.
    level : int
        Level at which to collapse nodes.  If the level specified
        is higher than the shortest root to leaf path in the tree,
        then that leaf will only be collapsed down to the length
        of its path.

    Returns
    -------
    trimmed_tree : skbio.TreeNode
        The collapsed skbio.TreeNode object.
    trimmed_table : pd.DataFrame
        The collapsed contingency table.

    Examples
    --------
    >>> from skbio import TreeNode
    >>> import pandas as pd
    >>> from canvas.tree import collapse
    >>> table = pd.DataFrame({'a': [10, 20, 30],
    ...                       'b': [5, 15, 25],
    ...                       'd': [1, 2, 3]},
    ...                      index=['s1', 's2', 's3'])
    >>> tree_str = u"((a,b)c,d);"
    >>> tree = TreeNode.read([tree_str])
    >>> print(tree.ascii_art())
                        /-a
              /c-------|
    ---------|          \-b
             |
              \-d

    If we want to collapse the first level of the tree, we can only collapse
    leaves `a` and `b` together.  These leaves will be combined into the
    internal node `c`, and their corresponding counts will be added together.
    The abundances of `d` will not be collapsed, since node `c` will need to
    be collapsed before `d` can be merged with `c`.

    >>> new_tree, new_table = collapse(tree, table, level=1)
    >>> print(new_tree.ascii_art())
              /-c
    ---------|
              \-d
    >>> new_table
        d   c
    s1  1  15
    s2  2  35
    s3  3  55
    """
    # Strategy : Identify all of the internal node with children as leaves
    # -  If both children are leaves - collapse those leaves together
    #    and collapse the corresponding counts in the table
    # -  Repeat for each level.
    counter = level
    trimmed_tree = tree.copy()
    trimmed_table = table.copy()

    while counter > 0:
        trimmed_tree, trimmed_table = _trim_level(trimmed_tree, trimmed_table)
        counter = counter - 1
    return trimmed_tree, trimmed_table
