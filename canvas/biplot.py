from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from skbio.stats.composition import clr, closure, centralize,
from skbio.stats.composition import multiplicative_replacement, perturb_inv
from biom import load_table
from functools import partial
import math


def make_biplot(sample_df,
                feature_df,
                sample_df_metadata=None,
                feature_df_metadata=None,
                sample_color_category=None,
                feature_color_category=None,
                sample_color_dict=None,
                feature_color_dict=None,
                eigenvalue=None):

    if not isinstance(sample_df, pd.core.frame.DataFrame):
        raise ValueError('`sample_df` must be a `pd.DataFrame`, '
                         'not %r.' % type(sample_df).__name__)

    if not isinstance(feature_df, pd.core.frame.DataFrame):
        raise ValueError('`feature_df` must be a `pd.DataFrame`, '
                         'not %r.' % type(feature_df).__name__)

    if sample_df_metadata is not None:
        if not isinstance(sample_df_metadata, pd.core.frame.DataFrame):
            raise ValueError('`sample_df_metadata` must be a `pd.DataFrame`, '
                             'not %r.' % type(sample_df_metadata).__name__)

    if feature_df_metadata is not None:
        if not isinstance(feature_df_metadata, pd.core.frame.DataFrame):
            raise ValueError('`feature_df_metadata` must be a `pd.DataFrame`, '
                             'not %r.' % type(feature_df_metadata).__name__)

    # if sample_color_dict is None.
    # Specify a default palette (default colors).
    # if (sample_color_dict == None)

    if sample_color_dict is not None:
        if not isinstance(sample_color_dict, dict):
            raise ValueError('`sample_color_dict` must be a `dictionary`, '
                             'not %r.' % type(sample_color_dict).__name__)

    # if feature_color_dict is None.
    # Specify a default palette (default colors).
    # if (feature_color_dict == None):
    if feature_color_dict is not None:
        if not isinstance(feature_color_dict, dict):
            raise ValueError('`feature_color_dict` must be a `dictionary`, '
                             'not %r.' % type(feature_color_dict).__name__)

    # check to see if pandas data frames are out of order,
    # do sort here and re-check if data frames are different lengths?
    if sample_df_metadata is not None:
        if (sample_df.index != sample_df_metadata.index).any():
            sample_df = sample_df.sort_index(axis=0)
            sample_df_metadata = sample_df_metadata.sort_index(axis=0)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    if sample_df_metadata is None:
        ax.plot(np.ravel(sample_df.ix[:, 0]), np.ravel(sample_df.ix[:, 1]),
                marker='o', linestyle='', ms=8)
    else:
        for name, group in sample_df.groupby(sample_df_metadata[
                                             samp_col_category]):
            if sample_color_dict is not None:
                ax.plot(np.ravel(group.ix[:, 0]), np.ravel(group.ix[:, 1]),
                        marker='o', linestyle='', ms=8,
                        color=sample_color_dict[name],
                        label=name)
            else:
                ax.plot(np.ravel(group.ix[:, 0]), np.ravel(group.ix[:, 1]),
                        marker='o', linestyle='', ms=8,
                        label=name)
    ax2 = ax.twinx()
    if sample_df_metadata is not None:
        ax.legend(title=sample_color_category, loc=2, numpoints=1)
    ax2.set_ylim(ax.get_ylim())

    recs = []
    feature = []
    otu_feature_color = []
    otu_feature_append = []

    # choose percent of arrows to show, feature,
    # which arrows and labeling of arrows?
    # feature_df =
    # feature_df[feature_df['radius'] >= max(feature_df['radius'])*.001]

    if feature_df_metadata is None:
        for index, row in feat_df.iterrows():
            ax2.arrow(0, 0, np.asscalar(row[0:1]), np.asscalar(row[1:2]),
                      width=0.02, head_width=0.05, alpha=0.7)
    else:
        for name, group in
        feature_df.groupby(feature_df_metadata[feature_color_category]):
            if feature_color_dict is not None:
                if len(name.split(';')) > 1:
                    otu_feature = name.split(';')[1].replace(' ', '')
                    otu_feature_color.append(feature_color_dict[otu_feature])
                    otu_feature_append.append(otu_feature)
                    ax2.arrow(0, 0, np.asscalar(group.ix[:, 0]),
                              np.asscalar(group.ix[:, 1]),
                              width=0.02, head_width=0.05, alpha=0.5,
                              color=feature_color_dict[otu_feature])
                else:
                    otu_feature = name
                    otu_feature_color.append(feature_color_dict[otu_feature])
                    otu_feature_append.append(otu_feature)
                    ax2.arrow(0, 0, np.asscalar(group.ix[:, 0]),
                              np.asscalar(group.ix[:, 1]),
                              width=0.02, head_width=0.05, alpha=0.5,
                              color=feature_color_dict[otu_feature])
            else:
                ax2.arrow(0, 0, np.asscalar(group.ix[:, 0]),
                          np.asscalar(group.ix[:, 1]),
                          width=0.02, head_width=0.05)

        color_otu_tax_df = pd.DataFrame({
                                            feature_color_category:
                                            np.array(otu_feature_append),
                                            'Color':
                                            np.array(otu_feature_color)
                                        })
        color_otu_tax_df = color_otu_tax_df.drop_duplicates()

        for name, group in color_otu_tax_df.groupby(feature_color_category):
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                        fc=feature_color_dict[name]))
            feature.append(name)
        if feature_color_category is not None:
            ax2.legend(recs, feature, loc=1, title=feature_color_category)

    xmin = min([min(sample_df.ix[:, 0]), min(feature_df.ix[:, 0])])
    xmax = max([max(sample_df.ix[:, 0]), max(feature_df.ix[:, 0])])
    ymin = min([min(sample_df.ix[:, 1]), min(feature_df.ix[:, 1])])
    ymax = max([max(sample_df.ix[:, 1]), max(feature_df.ix[:, 1])])
    xpad = (xmax - xmin) * 0.2
    ypad = (ymax - ymin) * 0.2

    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, xmax + ypad)
    ax2.set_xlim(xmin - xpad, xmax + xpad)
    ax2.set_ylim(ymin - ypad, xmax + ypad)
    ax2.set_yticks([])

    if eigenvalue is not None:
        ax.set_xlabel(
            'PC 1 ({:.2%})'.format(eigenvalue[0]**2/sum(eigenvalue**2))
        )
        ax.set_ylabel(
            'PC 2 ({:.2%})'.format(eigenvalue[1]**2/sum(eigenvalue**2))
        )

    return fig, [ax, ax2]
