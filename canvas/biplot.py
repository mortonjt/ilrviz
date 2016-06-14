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


    fig, ax = plt.subplots(figsize=(10,10))
    recs = []
    phyla= []

    for name, group in sample_df_test.groupby(sample_df_test_metadata[sample_color_category]):
        ax.plot(np.ravel(group.ix[:,0]), np.ravel(group.ix[:,1]),
                marker='o', linestyle='', ms=8,
                color = sample_color_dictionary[name],
                label = name)

    ax2 = ax.twinx()
    ax.legend(title=sample_df_test_metadata.dtypes.index[0], loc=2, numpoints = 1)
    ax2.set_ylim(ax.get_ylim())

    for name, group in feature_df_test.groupby(feature_df_test_metadata[feature_color_category]):
        ax2.arrow(0,0,np.asscalar(group.ix[:,0]), np.asscalar(group.ix[:,1]),
                  width = 0.02, head_width = 0.05,
                  color=feature_color_dictionary[name])
        recs.append(mpatches.Rectangle((0,0),1,1,fc=feature_color_dictionary[name]))
        phyla.append(name)
    ax2.legend(recs,phyla,loc=1)

    xmin = min([min(sample_df_test.ix[:,0]), min(feature_df_test.ix[:,0])])
    xmax = max([max(sample_df_test.ix[:,0]), max(feature_df_test.ix[:,0])])
    ymin = min([min(sample_df_test.ix[:,1]), min(feature_df_test.ix[:,1])])
    ymax = max([max(sample_df_test.ix[:,1]), max(feature_df_test.ix[:,1])])
    xpad = (xmax - xmin) * 0.3
    ypad = (ymax - ymin) * 0.3

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, xmax + ypad)
    ax2.set_xlim(xmin - xpad, xmax + xpad)
    ax2.set_ylim(ymin - ypad, xmax + ypad)
    ax2.set_yticks([])

    return fig, [ax, ax2]
