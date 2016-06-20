from __future__ import division
import unittest
from canvas.kernel.unifrac import aitchison_unifrac, pairwise_aitchison_unifrac
# from canvas.kernel.unifrac._unifrac import _fast_pairwise_weighted_logaitchison
from canvas.kernel.unifrac._unifrac import _pairwise_aitchison_unifrac_unrolled

import numpy.testing as npt
import numpy as np
import pandas as pd
from skbio import DistanceMatrix, TreeNode


class TestAitchisonUnifrac(unittest.TestCase):
    def setUp(self):
        pass

    def test_aitchison_unifrac_base_case(self):
        x = [1/2, 1/2]
        y = [1/4, 3/4]
        dm = np.array([[0, 1/2],
                       [1/2, 0]])
        res = aitchison_unifrac(x, y, dm)
        exp = np.sqrt((1/4) * (-np.log(1/3))**2)
        npt.assert_allclose(res, exp)

    def test_aitchison_unifrac(self):
        x = [1/2, 1/4, 1/4]
        y = [1/3, 1/3, 1/3]
        dm = np.array([[0, 1/2, 1/3],
                       [1/2, 0, 1/4],
                       [1/3, 1/4, 0]])
        res = aitchison_unifrac(x, y, dm)
        exp = np.sqrt((1/3) * ((1/2)*np.log(2)**2 + (1/3)*np.log(2)**2))
        npt.assert_allclose(res, exp)

    def test_pairwise_aitchison_unifrac(self):
        t = TreeNode.read([u"(a:1,(e:1, d:1)b:1)c;"])

        X = pd.DataFrame([[6, 3, 3], [4, 4, 4]],
                         index=['sample1', 'sample2'],
                         columns=['a', 'e', 'd'])

        res = pairwise_aitchison_unifrac(X, t)
        d = np.sqrt((1/3) * (3*np.log(2)**2 + 3*np.log(2)**2))
        exp = DistanceMatrix([[0, d], [d, 0]], ids=['sample1', 'sample2'])
        npt.assert_allclose(res.data, exp.data)
        self.assertEqual(res.ids, exp.ids)

    def test_pairwise_aitchison_unifrac_scrambled(self):
        t = TreeNode.read([u"(a:1,(e:1, d:1)b:1)c;"])

        X = pd.DataFrame([[6, 3, 3], [4, 4, 4]],
                         index=['sample1', 'sample2'],
                         columns=['a', 'd', 'e'])

        res = pairwise_aitchison_unifrac(X, t)
        d = np.sqrt((1/3) * (3*np.log(2)**2 + 3*np.log(2)**2))
        exp = DistanceMatrix([[0, d], [d, 0]], ids=['sample1', 'sample2'])
        npt.assert_allclose(res.data, exp.data)
        self.assertEqual(res.ids, exp.ids)

    # def test_fast_pairwise_aitchison_unifrac(self):
    #     N = 2
    #     D = 3
    #     dm = np.random.rand(D, D)**2
    #     dm = dm + dm.T
    #     X = abs(np.random.rand(N, D)*100) + 1
    #     out = np.zeros((N, N))
    #     exp = np.zeros((N, N))

    #     _fast_pairwise_weighted_logaitchison(np.log(X),
    #                                          dm, out)
    #     res = out + out.T
    #     _pairwise_aitchison_unifrac_unrolled(np.log(X),
    #                                          dm, exp)
    #     exp = exp + exp.T
    #     npt.assert_allclose(exp, res)

    def test_pairwise_aitchison_unifrac_unrolled(self):
        N = 3
        D = 2

        dm = np.array([[0, 1/2],
                       [1/2, 0]])
        X = np.array([[6, 6],
                      [9, 3],
                      [8, 4]])
        exp = np.array([[0, (1/2)*np.log(3)**2, (1/2)*np.log(2)**2],
                        [(1/2)*np.log(3)**2, 0,
                         (1/2)*(np.log(3)-np.log(2))**2],
                        [(1/2)*np.log(2)**2,
                         (1/2)*(np.log(3)-np.log(2))**2, 0]])
        exp = np.sqrt(exp/D)
        out = np.zeros((N, N))

        _pairwise_aitchison_unifrac_unrolled(np.log(X), dm, out)
        res = out + out.T
        npt.assert_allclose(exp, res)

if __name__ == '__main__':
    unittest.main()
