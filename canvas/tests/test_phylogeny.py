from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import numpy.testing as npt
from canvas.phylogeny import phylogenetic_basis, _count_matrix, _balance_basis
from skbio import TreeNode
from skbio.util import get_data_path


class TestPhylogeny(unittest.TestCase):
    def setUp(self):
        pass

    def test_count_matrix_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        res = _count_matrix(t)
        exp = {'k': 0, 'l': 1, 'r': 1, 't': 0, 'tips': 2}
        self.assertEqual(res[t], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[0]], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[1]], exp)

    def test_count_matrix_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        res = _count_matrix(t)

        exp = {'k': 0, 'l': 2, 'r': 1, 't': 0, 'tips': 3}
        self.assertEqual(res[t], exp)

        exp = {'k': 1, 'l': 1, 'r': 1, 't': 0, 'tips': 2}
        self.assertEqual(res[t[0]], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[1]], exp)
        self.assertEqual(res[t[0][0]], exp)
        self.assertEqual(res[t[0][1]], exp)

    def test_count_matrix_singleton_error(self):
        with self.assertRaises(ValueError):
            tree = u"(((a,b)c, d)root);"
            t = TreeNode.read([tree])
            _count_matrix(t)

    def test_count_matrix_trifurcating_error(self):
        with self.assertRaises(ValueError):
            tree = u"((a,b,e)c, d);"
            t = TreeNode.read([tree])
            _count_matrix(t)

    def test_balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])

        exp_basis = np.array([[np.sqrt(1. / 2), -np.sqrt(1. / 2)]])
        exp_keys = [t]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertItemsEqual(exp_keys, res_keys)

    def test_balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])

        exp_basis = np.array([[np.sqrt(2. / 3), -np.sqrt(1. / 6),
                               -np.sqrt(1. / 6)],
                              [0, np.sqrt(1. / 2), -np.sqrt(1. / 2)]
                              ])
        exp_keys = [t, t[0]]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertItemsEqual(exp_keys, res_keys)

    def test_phylogenetic_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        exp_keys = [t]
        exp_basis = np.array([0.80442968, 0.19557032])
        res_basis, res_keys = phylogenetic_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertItemsEqual(exp_keys, res_keys)

    def test_phylogenetic_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        exp_keys = [t, t[0]]
        exp_basis = np.array([[0.62985567, 0.18507216, 0.18507216],
                             [0.28399541, 0.57597535, 0.14002925]])

        res_basis, res_keys = phylogenetic_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertItemsEqual(exp_keys, res_keys)

    def test_phylogenetic_basis_large1(self):
        fname = get_data_path('large_tree1.nwk',
                              subfolder='data/phylogeny')
        t = TreeNode.read(fname)
        exp_basis = np.loadtxt(
            get_data_path('large_tree1_basis.txt',
                          subfolder='data/phylogeny'))
        res_basis, res_keys = phylogenetic_basis(t)
        npt.assert_allclose(exp_basis, res_basis)

    def test_phylogenetic_basis_large2(self):
        fname = get_data_path('large_tree2.nwk',
                              subfolder='data/phylogeny')
        t = TreeNode.read(fname)
        exp_basis = np.loadtxt(
            get_data_path('large_tree2_basis.txt',
                          subfolder='data/phylogeny'))
        res_basis, res_keys = phylogenetic_basis(t)
        npt.assert_allclose(exp_basis, res_basis)

if __name__ == "__main__":
    unittest.main()
