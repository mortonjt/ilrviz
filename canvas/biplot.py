from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from skbio.stats.composition import clr, closure, centralize



def make_biplot(sample_df_test,
                feature_df_test,
                sample_df_test_metadata,
                feature_df_test_metadata,
                sample_color_category,
                sample_color_dictionary,
                feature_color_category,
                feature_color_dictionary):


    fig = plt.subplots(figsize=(10,10))
    recs = []
    phyla= []

    for name, group in sample_df_test.groupby(sample_df_test_metadata[sample_color_category]):
            plt.plot(np.ravel(group.ix[:,0]), np.ravel(group.ix[:,1]),
                       marker='o', linestyle='', ms=8,
                       color = sample_color_dictionary[name],
                       label = name)
            legend_1 = plt.legend(title=sample_df_test_metadata.dtypes.index[0], loc=2, numpoints = 1)
    plt.gca().add_artist(legend_1)

    for name, group in feature_df_test.groupby(feature_df_test_metadata[feature_color_category]):
        plt.arrow(0,0,np.asscalar(group.ix[:,0]), np.asscalar(group.ix[:,1]),
                    width = 0.02, head_width = 0.05,
                    color=feature_color_dictionary[name])
        recs.append(mpatches.Rectangle((0,0),1,1,fc=feature_color_dictionary[name]))
        phyla.append(name)
        plt.legend(recs,phyla,loc=1)

    xmin = min([min(feature_df_test.ix[:,0])])
    xmax = max([max(feature_df_test.ix[:,0])])
    ymin = min([min(feature_df_test.ix[:,0])])
    ymax = max([max(feature_df_test.ix[:,0])])
    xpad = (xmax - xmin) * 0.5
    ypad = (ymax - ymin) * 0.5
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(ymin - ypad, ymax + ypad)

    return fig
