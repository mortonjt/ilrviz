from __future__ import division
import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from skbio.stats.composition import ilr_inv, ilr
from canvas.stats.regression import simplicialOLS, _regression
from skbio.util import assert_data_frame_almost_equal


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.y = pd.DataFrame([[0.5, 0.5],
                               [0.52631579, 0.47368421],
                               [0.55, 0.45],
                               [0.57142857, 0.42857143],
                               [0.59090909, 0.40909091],
                               [0.60869565, 0.39130435],
                               [0.625, 0.375],
                               [0.64, 0.36],
                               [0.65384615, 0.34615385],
                               [0.66666667, 0.33333333]])
        self.x = pd.DataFrame([[1., 0.],
                               [1., 1.11111111],
                               [1., 2.22222222],
                               [1., 3.33333333],
                               [1., 4.44444444],
                               [1., 5.55555556],
                               [1., 6.66666667],
                               [1., 7.77777778],
                               [1., 8.88888889],
                               [1., 10.]])

    def test_simple_regression(self):
        _x = np.linspace(-1, 1, 10)
        x = np.vstack((np.ones(10), _x)).T
        ilr_y = np.linspace(-10, 10, 10)
        y = ilr_inv(ilr_y.reshape((10, 1)))
        predict, b, e, r2 = _regression(y, x)
        exp = np.array([0, 10.0])
        npt.assert_allclose(ilr(b), exp, rtol=0, atol=1e-8)
        npt.assert_allclose(r2, 1.0)

    def test_2D_regression(self):
        _x = np.linspace(-1, 1, 10)
        x = np.vstack((np.ones(10), _x)).T

        ilr_y = np.linspace(-10, 10, 10)
        ilr_y = np.vstack((ilr_y, ilr_y[::-1])).T
        y = ilr_inv(ilr_y)
        predict, b, e, r2 = _regression(y, x)
        exp = np.array([[0., 0.],
                        [10., -10.]])
        npt.assert_allclose(ilr(b), exp, rtol=0, atol=1e-8)
        npt.assert_allclose(r2, 1.0)

    def test_simple_simplicialOLS(self):
        _x = np.linspace(-1, 1, 10)
        x = pd.DataFrame(np.vstack((np.ones(10), _x)).T)
        ilr_y = np.linspace(-10, 10, 10)
        y = pd.DataFrame(ilr_inv(ilr_y.reshape((10, 1))))
        predict, b, e, r2 = simplicialOLS(y, x)
        exp = np.array([0, 10.0])
        npt.assert_allclose(ilr(b), exp, rtol=0, atol=1e-8)
        npt.assert_allclose(r2, 1.0)

        # For a 2D simplex, [0.5, 0.5] means no error
        exp_e = pd.DataFrame([[0.5, 0.5]]*10)
        exp_b = pd.DataFrame([[0.5, 0.5],
                              [0.999999,  7.213536e-7]],
                             index=['b0', 'b1'])
        assert_data_frame_almost_equal(exp_e, e)
        assert_data_frame_almost_equal(exp_b, b)
        assert_data_frame_almost_equal(predict, y)

    def test_2D_simplicialOLS_indexed(self):
        x = pd.DataFrame(
            [[1., -1.],
             [1., -0.77777778],
             [1., -0.55555556],
             [1., -0.33333333],
             [1., -0.11111111],
             [1., 0.11111111],
             [1., 0.33333333],
             [1., 0.55555556],
             [1., 0.77777778],
             [1., 1.]],
            columns=['const', 'pH'],
            index=['a', 'b', 'c', 'd', 'e',
                   'f', 'g', 'h', 'i', 'j'],
        )
        y = pd.DataFrame(
            [[7.21353629e-07, 9.99999275e-01, 4.07450222e-09],
             [1.67107919e-05, 9.99982991e-01, 2.98191523e-07],
             [3.86974837e-04, 9.99591210e-01, 2.18148843e-05],
             [8.87465868e-03, 9.89544844e-01, 1.58049735e-03],
             [1.56844234e-01, 7.54912293e-01, 8.82434734e-02],
             [3.34989515e-01, 6.95990438e-02, 5.95411442e-01],
             [1.50964509e-01, 1.35391387e-03, 8.47681578e-01],
             [5.33634592e-02, 2.06587610e-05, 9.46615882e-01],
             [1.75314085e-02, 2.92968701e-07, 9.82468299e-01],
             [5.61668624e-03, 4.05161995e-09, 9.94383310e-01]],
            index=['a', 'b', 'c', 'd', 'e',
                   'f', 'g', 'h', 'i', 'j'],
            columns=['OTU1', 'OTU2', 'OTU3']
        )
        predict, b, e, r2 = simplicialOLS(y, x)

        exp_e = pd.DataFrame([[1/3, 1/3, 1/3]]*10,
                             index=['a', 'b', 'c', 'd', 'e',
                                    'f', 'g', 'h', 'i', 'j'],
                             columns=['OTU1', 'OTU2', 'OTU3'])
        exp_b = np.array([[0, 0],
                          [10, -10]])
        exp_b = pd.DataFrame(ilr_inv(exp_b),
                             index=['b0', 'b1'],
                             columns=['OTU1', 'OTU2', 'OTU3'])
        assert_data_frame_almost_equal(exp_e, e)
        assert_data_frame_almost_equal(exp_b, b)
        assert_data_frame_almost_equal(predict, y)

    def test_2D_simplicialOLS_scrambled(self):
        x = pd.DataFrame(
            [[1., 1.],
             [1., -0.77777778],
             [1., -0.55555556],
             [1., -0.33333333],
             [1., -0.11111111],
             [1., 0.11111111],
             [1., 0.33333333],
             [1., 0.55555556],
             [1., 0.77777778],
             [1., -1.]],
            columns=['const', 'pH'],
            index=['j', 'b', 'c', 'd', 'e',
                   'f', 'g', 'h', 'i', 'a']
        )
        y = pd.DataFrame(
            [[7.21353629e-07, 9.99999275e-01, 4.07450222e-09],
             [1.67107919e-05, 9.99982991e-01, 2.98191523e-07],
             [3.86974837e-04, 9.99591210e-01, 2.18148843e-05],
             [8.87465868e-03, 9.89544844e-01, 1.58049735e-03],
             [1.56844234e-01, 7.54912293e-01, 8.82434734e-02],
             [3.34989515e-01, 6.95990438e-02, 5.95411442e-01],
             [1.50964509e-01, 1.35391387e-03, 8.47681578e-01],
             [5.33634592e-02, 2.06587610e-05, 9.46615882e-01],
             [1.75314085e-02, 2.92968701e-07, 9.82468299e-01],
             [5.61668624e-03, 4.05161995e-09, 9.94383310e-01]],
            index=['a', 'b', 'c', 'd', 'e',
                   'f', 'g', 'h', 'i', 'j'],
            columns=['OTU1', 'OTU2', 'OTU3']
        )
        predict, b, e, r2 = simplicialOLS(y, x)

        exp_e = pd.DataFrame([[1/3, 1/3, 1/3]]*10,
                             index=['a', 'b', 'c', 'd', 'e',
                                    'f', 'g', 'h', 'i', 'j'],
                             columns=['OTU1', 'OTU2', 'OTU3'])
        exp_b = np.array([[0, 0],
                          [10, -10]])
        exp_b = pd.DataFrame(ilr_inv(exp_b),
                             index=['b0', 'b1'],
                             columns=['OTU1', 'OTU2', 'OTU3'])
        exp_e = exp_e.reindex(index=['j', 'b', 'c', 'd', 'e',
                                     'f', 'g', 'h', 'i', 'a'])
        exp_y = y.reindex(index=['j', 'b', 'c', 'd', 'e',
                                 'f', 'g', 'h', 'i', 'a'])
        assert_data_frame_almost_equal(exp_e, e)
        assert_data_frame_almost_equal(exp_b, b)
        assert_data_frame_almost_equal(predict, exp_y)

    def test_bad_simplicalOLS_Y(self):
        with self.assertRaises(TypeError):
            simplicialOLS(self.y.values, self.x)

    def test_bad_simplicalOLS_X(self):
        with self.assertRaises(TypeError):
            simplicialOLS(self.y, self.x.values)

    def test_bad_simplicalOLS_Y_null(self):
        y = self.y
        y.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            simplicialOLS(y, self.x)

    def test_bad_simplicalOLS_X_null(self):
        x = self.y
        x.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            simplicialOLS(self.y, x)

    def test_bad_simplicalOLS_Y_zero(self):
        y = self.y
        y.iloc[0, 0] = 0
        with self.assertRaises(ValueError):
            simplicialOLS(y, self.x)


if __name__ == "__main__":
    unittest.main()
