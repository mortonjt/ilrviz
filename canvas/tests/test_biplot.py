from __future__ import absolute_import, division, print_function
import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from canvas.biplot import make_biplot

class TestBiplot(unittest.TestCase):
    def setUp(self):

        self.sample_df_test = pd.DataFrame({'PC1' : [-1.25, 0, -1.25, 0, 1.25, 1.25],
                                            'PC2' : [-1.25, -1.25, 0, 1.25, 0, 1.25]},
                                            index = ['S1','S2','S3', 'S4', 'S5', 'S6'])

        self.feature_df_test = pd.DataFrame({'PC1': [1,-1,1],
                                             'PC2': [1,-1,0]},
                                             index = ['OTU1', 'OTU2', 'OTU3'])

        self.sample_df_test_metadata = pd.DataFrame({'Days' : ['One','One','One','Two','Two','Two']},
                                                     index = ['S1','S2','S3', 'S4', 'S5', 'S6'])

        self.feature_df_test_metadata = pd.DataFrame({'Location': ['Oral','Gut','Skin']},
                                                      index = ['OTU1', 'OTU2', 'OTU3'])

        self.sample_color_dictionary = {'One': '#6CAD3F',
                                        'Two': '#CF5635'}

        self.feature_color_dictionary = {'Oral': '#219F8D',
                                         'Gut': '#D04984',
                                         'Skin': '#D4D71C'}
    def test_sample_position(self):

        fig = make_biplot(self.sample_df_test,
                          self.feature_df_test,
                          sample_df_test_metadata=self.sample_df_test_metadata,
                          feature_df_test_metadata=self.feature_df_test_metadata,
                          sample_color_category='Days',
                          sample_color_dictionary=self.sample_color_dictionary,
                          feature_color_category='Location',
                          feature_color_dictionary=self.feature_color_dictionary)
        f, a = fig
        expected = np.array([[-1.25, -1.25],
                             [ 0.  , -1.25],
                             [-1.25,  0.  ]])
        npt.assert_allclose(expected, a.lines[0].get_xydata())

    def test_sample_color(self):
        pass


if __name__=='__main__':
    unittest.main()
