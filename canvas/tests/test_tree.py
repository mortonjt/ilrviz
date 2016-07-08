from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import pandas as pd
from canvas.tree import collapse, _trim_level, _is_collapsible
from skbio import TreeNode

from skbio.util import assert_data_frame_almost_equal


class TestTree(unittest.TestCase):

    def test__is_collapsible(self):
        tree_str = u"((a,b)c,d);"
        tree = TreeNode.read([tree_str])
        self.assertFalse(_is_collapsible(tree))
        self.assertTrue(_is_collapsible(tree.children[0]))
        self.assertFalse(_is_collapsible(tree.children[1]))
        self.assertFalse(_is_collapsible(tree.children[0].children[0]))
        self.assertFalse(_is_collapsible(tree.children[0].children[1]))

    def test__trim_level_base_case(self):
        tree_str = u"(a,b)c;"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[15, 35, 55]]).T,
            index=['s1', 's2', 's3'],
            columns=['c'])
        exp_tree = TreeNode.read([u"c;"])

        res_tree, res_table = _trim_level(tree, table)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())
        assert_data_frame_almost_equal(exp_table, res_table)

    def test__trim_level(self):
        tree_str = u"((a,b)c,d);"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25],
                              'd': [1, 2, 3]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[1, 2, 3],
                      [15, 35, 55]]).T,
            index=['s1', 's2', 's3'],
            columns=['d', 'c'])
        exp_tree = TreeNode.read([u"(c,d);"])

        res_tree, res_table = _trim_level(tree, table)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())
        assert_data_frame_almost_equal(exp_table, res_table)

    def test__trim_level_no_name(self):
        tree_str = u"((a,b), d);"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25],
                              'd': [1, 2, 3]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[1, 2, 3],
                      [15, 35, 55]]).T,
            index=['s1', 's2', 's3'],
            columns=['d', 1])

        exp_tree = TreeNode.read([u"(,d);"])

        res_tree, res_table = _trim_level(tree, table)

        self.assertEqual(str(exp_tree), str(res_tree))
        assert_data_frame_almost_equal(exp_table, res_table)

    def test_trifurcating_base_case(self):
        tree_str = u"(a,b,c)d;"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25],
                              'c': [25, 15, 25]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[40, 50, 80]]).T,
            index=['s1', 's2', 's3'],
            columns=['d'])
        exp_tree = TreeNode.read([u"d;"])

        res_tree, res_table = _trim_level(tree, table)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())
        assert_data_frame_almost_equal(exp_table, res_table)

    def test_trifurcating_single_child(self):
        tree_str = u"((((a,b)c,(f)d),g)r);"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25],
                              'f': [25, 15, 25],
                              'g': [25, 15, 25]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[25, 15, 25],
                      [15, 35, 55],
                      [25, 15, 25]]).T,
            index=['s1', 's2', 's3'],
            columns=['g', 'c', 'd'])
        exp_tree = TreeNode.read([u"(((c,d), g)r);"])

        res_tree, res_table = _trim_level(tree, table)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())
        assert_data_frame_almost_equal(exp_table, res_table)

    def test_collapse(self):
        # Collapse 1 level
        tree_str = u"((a,b)c, d);"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25],
                              'd': [1, 2, 3]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[1, 2, 3],
                      [15, 35, 55]]).T,
            index=['s1', 's2', 's3'],
            columns=['d', 'c'])
        exp_tree = TreeNode.read([u"(c,d);"])

        res_tree, res_table = collapse(tree, table, 1)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())
        assert_data_frame_almost_equal(exp_table, res_table)

    def test_collapse_2(self):
        # Collapse 2 levels
        tree_str = u"((a,b)c, d);"
        tree = TreeNode.read([tree_str])
        table = pd.DataFrame({'a': [10, 20, 30],
                              'b': [5, 15, 25],
                              'd': [1, 2, 3]},
                             index=['s1', 's2', 's3'])
        exp_table = pd.DataFrame(
            np.array([[16, 37, 58]]).T,
            index=['s1', 's2', 's3'],
            columns=[0])
        exp_tree = TreeNode.read([u";"])

        res_tree, res_table = collapse(tree, table, 2)
        self.assertEqual(exp_tree.ascii_art(), res_tree.ascii_art())
        assert_data_frame_almost_equal(exp_table, res_table)

    def test_collapse_no_table(self):
        pass

if __name__ == '__main__':
    unittest.main()
