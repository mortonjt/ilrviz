from __future__ import absolute_import, division, print_function
import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
from canvas.biplot import make_biplot


class TestBiplot(unittest.TestCase):
    def setUp(self):

        self.sample_test = pd.DataFrame(
            {'PCA1': [-1.25, 0, -1.25, 0, 1.25, 1.25],
             'PCA2': [-1.25, -1.25, 0, 1.25, 0, 1.25]},
            index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

        self.feature_test = pd.DataFrame(
            {'PCA1': [1.25, -1.25, 1.25],
             'PCA2': [1.25, -1.25, 0]},
            index=['OTU1', 'OTU2', 'OTU3'])

        self.sample_test_big = pd.DataFrame(
            {'PCA1': [-3, 0, -3, 0, 3, 3],
             'PCA2': [-3, -3, 0, 3, 0, 3]},
            index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

        self.feature_test_small = pd.DataFrame(
            {'PCA1': [0.5, -0.5, .5],
             'PCA2': [0.5, -.5, 0]},
            index=['OTU1', 'OTU2', 'OTU3'])

        self.sample_test_small = pd.DataFrame(
            {'PCA1': [-.5, 0, -.5, 0, .5, .5],
             'PCA2': [-.5, -.5, 0, .5, 0, .5]},
            index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

        self.feature_test_big = pd.DataFrame(
            {'PCA1': [3, -3, 3],
             'PCA2': [3, -3, 0]},
            index=['OTU1', 'OTU2', 'OTU3'])

        self.sample_test_metadata = pd.DataFrame(
            {'Days': ['One', 'One', 'One', 'Two', 'Two', 'Two']},
            index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

        self.sample_test_metadata_mx = pd.DataFrame(
            {'Days': ['One', 'One', 'One', 'Two', 'Two', 'Two']},
            index=['S1', 'S2', 'S3', 'S7', 'S5', 'S6'])

        self.feature_test_metadata = pd.DataFrame(
            {'Loc': ['Oral', 'Gut', 'Skin']},
            index=['OTU1', 'OTU2', 'OTU3'])

        self.sample_color_test = pd.Series(
                ['#6CAD3F', '#CF5635'],
                index=['One', 'Two'])

        self.feature_color_test = pd.Series(
                ['#219F8D', '#D04984', '#D4D71C'],
                index=['Oral', 'Gut', 'Skin'])

    def test_sample_position(self):

        fig = make_biplot(self.sample_test,
                          features=self.feature_test,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_sample_1 = np.array([[-1.25, -1.25],
                                      [0., -1.25],
                                      [-1.25, 0.]])

        expected_sample_2 = np.array([[0., 1.25],
                                      [1.25, 0.],
                                      [1.25, 1.25]])

        npt.assert_allclose(expected_sample_1, a[0].lines[0].get_xydata())
        npt.assert_allclose(expected_sample_2, a[0].lines[1].get_xydata())

    def test_sample_position_big(self):

        fig = make_biplot(self.sample_test_big,
                          features=self.feature_test_small,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_sample_1 = np.array([[-3., -3.],
                                      [0., -3.],
                                      [-3., 0.]])

        expected_sample_2 = np.array([[0., 3.],
                                      [3., 0.],
                                      [3., 3.]])

        npt.assert_allclose(expected_sample_1, a[0].lines[0].get_xydata())
        npt.assert_allclose(expected_sample_2, a[0].lines[1].get_xydata())

    def test_sample_position_small(self):

        fig = make_biplot(self.sample_test_small,
                          features=self.feature_test_big,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_sample_1 = np.array([[-0.5, -0.5],
                                      [0., -0.5],
                                      [-0.5, 0.]])

        expected_sample_2 = np.array([[0., 0.5],
                                      [0.5, 0.],
                                      [0.5, 0.5]])

        npt.assert_allclose(expected_sample_1, a[0].lines[0].get_xydata())
        npt.assert_allclose(expected_sample_2, a[0].lines[1].get_xydata())

    def test_feature_arrow(self):

        fig = make_biplot(self.sample_test,
                          features=self.feature_test,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_arrow1 = np.array([[-1.30303301, -1.30303301],
                                    [-1.26767767, -1.23232233],
                                    [-1.25707107, -1.24292893],
                                    [-0.00707107, 0.00707107],
                                    [0.00707107, -0.00707107],
                                    [-1.24292893, -1.25707107],
                                    [-1.23232233, -1.26767767],
                                    [-1.30303301, -1.30303301]])

        expected_arrow2 = np.array([[1.30303301, 1.30303301],
                                    [1.26767767, 1.23232233],
                                    [1.25707107, 1.24292893],
                                    [0.00707107, -0.00707107],
                                    [-0.00707107, 0.00707107],
                                    [1.24292893, 1.25707107],
                                    [1.23232233, 1.26767767],
                                    [1.30303301, 1.30303301]])

        expected_arrow3 = np.array([[1.325, 0.],
                                    [1.25, -0.025],
                                    [1.25, -0.01],
                                    [0., -0.01],
                                    [0., 0.01],
                                    [1.25, 0.01],
                                    [1.25, 0.025],
                                    [1.325, 0.]])

        npt.assert_allclose(expected_arrow1, a[1].artists[0].xy, atol=.0001)
        npt.assert_allclose(expected_arrow2, a[1].artists[1].xy, atol=.0001)
        npt.assert_allclose(expected_arrow3, a[1].artists[2].xy, atol=.0001)

    def test_feature_arrow_small(self):

        fig = make_biplot(self.sample_test_big,
                          features=self.feature_test_small,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_arrow1 = np.array([[-0.55303301, -0.55303301],
                                    [-0.51767767, -0.48232233],
                                    [-0.50707107, -0.49292893],
                                    [-0.00707107, 0.00707107],
                                    [0.00707107, -0.00707107],
                                    [-0.49292893, -0.50707107],
                                    [-0.48232233, -0.51767767],
                                    [-0.55303301, -0.55303301]])

        expected_arrow2 = np.array([[0.55303301, 0.55303301],
                                    [0.51767767, 0.48232233],
                                    [0.50707107, 0.49292893],
                                    [0.00707107, -0.00707107],
                                    [-0.00707107, 0.00707107],
                                    [0.49292893, 0.50707107],
                                    [0.48232233, 0.51767767],
                                    [0.55303301, 0.55303301]])

        expected_arrow3 = np.array([[5.75000000e-01, 0.00000000e+00],
                                    [5.00000000e-01, -2.50000000e-02],
                                    [5.00000000e-01, -1.00000000e-02],
                                    [5.55111512e-17, -1.00000000e-02],
                                    [5.55111512e-17, 1.00000000e-02],
                                    [5.00000000e-01, 1.00000000e-02],
                                    [5.00000000e-01, 2.50000000e-02],
                                    [5.75000000e-01, 0.00000000e+00]])

        npt.assert_allclose(expected_arrow1, a[1].artists[0].xy, atol=.0001)
        npt.assert_allclose(expected_arrow2, a[1].artists[1].xy, atol=.0001)
        npt.assert_allclose(expected_arrow3, a[1].artists[2].xy, atol=.0001)

    def test_feature_arrow_big(self):

        fig = make_biplot(self.sample_test_small,
                          features=self.feature_test_big,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_arrow1 = np.array([[-3.05303301, -3.05303301],
                                    [-3.01767767, -2.98232233],
                                    [-3.00707107, -2.99292893],
                                    [-0.00707107, 0.00707107],
                                    [0.00707107, -0.00707107],
                                    [-2.99292893, -3.00707107],
                                    [-2.98232233, -3.01767767],
                                    [-3.05303301, -3.05303301]])

        expected_arrow2 = np.array([[3.05303301,  3.05303301],
                                    [3.01767767,  2.98232233],
                                    [3.00707107,  2.99292893],
                                    [0.00707107, -0.00707107],
                                    [-0.00707107,  0.00707107],
                                    [2.99292893,  3.00707107],
                                    [2.98232233,  3.01767767],
                                    [3.05303301,  3.05303301]])

        expected_arrow3 = np.array([[3.075,  0.],
                                    [3., -0.025],
                                    [3., -0.01],
                                    [0., -0.01],
                                    [0., 0.01],
                                    [3., 0.01],
                                    [3., 0.025],
                                    [3.075, 0.]])

        npt.assert_allclose(expected_arrow1, a[1].artists[0].xy,
                            atol=.0001)
        npt.assert_allclose(expected_arrow2, a[1].artists[1].xy,
                            atol=.0001)
        npt.assert_allclose(expected_arrow3, a[1].artists[2].xy,
                            atol=.0001)

    def test_x_y_limits(self):

        fig = make_biplot(self.sample_test,
                          features=self.feature_test,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_axis_1_xlim = (-2.0, 2.0)
        expected_axis_1_ylim = (-2.0, 2.0)
        expected_axis_2_xlim = (-2.0, 2.0)
        expected_axis_2_ylim = (-2.0, 2.0)

        npt.assert_allclose(expected_axis_1_xlim, a[0].get_xlim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_1_ylim, a[0].get_ylim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_2_xlim, a[1].get_xlim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_2_ylim, a[1].get_ylim(),
                            atol=.0001)

    def test_x_y_limits_sample_big(self):

        fig = make_biplot(self.sample_test_big,
                          features=self.feature_test_small,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_axis_1_xlim = (-4.7999999999999998, 4.7999999999999998)
        expected_axis_1_ylim = (-4.7999999999999998, 4.7999999999999998)
        expected_axis_2_xlim = (-4.7999999999999998, 4.7999999999999998)
        expected_axis_2_ylim = (-4.7999999999999998, 4.7999999999999998)

        npt.assert_allclose(expected_axis_1_xlim, a[0].get_xlim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_1_ylim, a[0].get_ylim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_2_xlim, a[1].get_xlim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_2_ylim, a[1].get_ylim(),
                            atol=.0001)

    def test_x_y_limits_sample_small(self):

        fig = make_biplot(self.sample_test_small,
                          features=self.feature_test_big,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        expected_axis_1_xlim = (-4.7999999999999998, 4.7999999999999998)
        expected_axis_1_ylim = (-4.7999999999999998, 4.7999999999999998)
        expected_axis_2_xlim = (-4.7999999999999998, 4.7999999999999998)
        expected_axis_2_ylim = (-4.7999999999999998, 4.7999999999999998)

        npt.assert_allclose(expected_axis_1_xlim, a[0].get_xlim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_1_ylim, a[0].get_ylim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_2_xlim, a[1].get_xlim(),
                            atol=.0001)
        npt.assert_allclose(expected_axis_2_ylim, a[1].get_ylim(),
                            atol=.0001)

    def test_sample_metadata_grouping_label_category(self):

        fig = make_biplot(self.sample_test,
                          features=self.feature_test,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        test_sample_metadata_grouping_label_category1 = ("One")
        test_sample_metadata_grouping_label_category2 = ("Two")

        npt.assert_equal(test_sample_metadata_grouping_label_category1,
                         a[0].legend_.legendHandles[0].get_label())
        npt.assert_equal(test_sample_metadata_grouping_label_category2,
                         a[0].legend_.legendHandles[1].get_label())

    def test_sample_metadata_grouping_color_category(self):

        fig = make_biplot(self.sample_test,
                          features=self.feature_test,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        test_sample_metadata_grouping_color_category1 = ("#6CAD3F")
        test_sample_metadata_grouping_color_category2 = ("#CF5635")

        npt.assert_equal(test_sample_metadata_grouping_color_category1,
                         a[0].legend_.legendHandles[0].get_color())
        npt.assert_equal(test_sample_metadata_grouping_color_category2,
                         a[0].legend_.legendHandles[1].get_color())

    def test_feature_metadata_grouping_label_category(self):
        fig = make_biplot(self.sample_test,
                          features=self.feature_test,
                          sample_metadata=self.sample_test_metadata['Days'],
                          feature_metadata=self.feature_test_metadata['Loc'],
                          sample_color=self.sample_color_test,
                          feature_color=self.feature_color_test)
        f, a = fig

        test_feature_metadata_grouping_label_category1 = ("Oral")
        test_feature_metadata_grouping_label_category2 = ("Gut")
        test_feature_metadata_grouping_label_category3 = ("Skin")

        npt.assert_equal(test_feature_metadata_grouping_label_category1,
                         a[1].legend_.texts[0].get_text())
        npt.assert_equal(test_feature_metadata_grouping_label_category2,
                         a[1].legend_.texts[1].get_text())
        npt.assert_equal(test_feature_metadata_grouping_label_category3,
                         a[1].legend_.texts[2].get_text())

    def test_sample_and_metadata_index_matching(self):
        with self.assertRaises(ValueError):
            make_biplot(self.sample_test,
                        sample_metadata=self.sample_test_metadata_mx['Days'])

    def test_sample_warning(self):
        with self.assertRaises(ValueError):
            make_biplot([])

if __name__ == '__main__':
    unittest.main()
