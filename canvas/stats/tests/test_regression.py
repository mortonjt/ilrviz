import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from skbio.stats.composition import power, ilr_inv, ilr
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
        predict, b, e, r2  = _regression(y, x)
        exp = np.array([0, 10.0])
        npt.assert_allclose(ilr(b), exp, rtol=0, atol=1e-8)
        npt.assert_allclose(r2, 1.0)

    def test_simple_simplicialOLS(self):
        _x = np.linspace(-1, 1, 10)
        x = pd.DataFrame(np.vstack((np.ones(10), _x)).T)
        ilr_y = np.linspace(-10, 10, 10)
        y = pd.DataFrame(ilr_inv(ilr_y.reshape((10, 1))))

        predict, b, e, r2  = simplicialOLS(y, x)
        exp = np.array([0, 10.0])
        npt.assert_allclose(ilr(b), exp, rtol=0, atol=1e-8)
        npt.assert_allclose(r2, 1.0)

        # For a 2D simplex, [0.5, 0.5] means no error
        exp_e = pd.DataFrame([[0.5, 0.5]]*10)
        exp_b = pd.DataFrame([[0.5, 0.5],
                              [0.999999,  7.213536e-7]],
                             columns=['b0', 'b1'])
        assert_data_frame_almost_equal(exp_e, e)
        assert_data_frame_almost_equal(exp_b, b)
        assert_data_frame_almost_equal(predict, y)

    def test_2D_simplicialOLS(self):
        _x = np.linspace(-1, 1, 10)
        x = np.vstack((np.ones(10), _x)).T

        ilr_y = np.linspace(-10, 10, 10)
        ilr_y = np.vstack((ilr_y, ilr_y[::-1])).T
        y = ilr_inv(ilr_y)
        predict, b, e, r2  = _regression(y, x)
        exp = np.array([[0, 0],
                        [10, -10]])
        npt.assert_allclose(ilr(b), exp, rtol=0, atol=1e-8)
        npt.assert_allclose(r2, 1.0)

    def test_bad_simplicalOLS_Y(self):
        with self.assertRaises(TypeError):
            res = simplicialOLS(self.y.values, self.x)

    def test_bad_simplicalOLS_X(self):
        with self.assertRaises(TypeError):
            res = simplicialOLS(self.y, self.x.values)

    def test_bad_simplicalOLS_Y_null(self):
        y = self.y
        y.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            res = simplicialOLS(y, self.x)

    def test_bad_simplicalOLS_X_null(self):
        x = self.y
        x.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            res = simplicialOLS(self.y, x)

    def test_bad_simplicalOLS_Y_zero(self):
        y = self.y
        y.iloc[0, 0] = 0
        with self.assertRaises(ValueError):
            res = simplicialOLS(y, self.x)

if __name__=="__main__":
    unittest.main()
