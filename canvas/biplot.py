from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
from collections import OrderedDict
import collections
import math


def make_biplot(sample_df,
                feature_df=None,
                sample_df_metadata=None,
                feature_df_metadata=None,
                sample_color_category=None,
                feature_color_category=None,
                sample_color_dict=None,
                feature_color_dict=None,
                **kwargs):

    figure_size = (15, 15)
    sample_df_x = 'PCA1'
    sample_df_y = 'PCA2'
    samp_col = 'RdGy'
    samp_alpha = 1
    samp_marker = 'o'
    samp_ms = 8
    samp_leg_loc = 2
    feature_df_x = 'PCA1'
    feature_df_y = 'PCA2'
    feat_col = 'Set1'
    feat_alpha = 1
    arrow_width = 0.02
    arrow_head = 0.05
    taxa_rank = 1
    feat_leg_loc = 1
    feature_order = 0
    sample_drop_list = []
    show_color_drop = False
    sample_drop_col = ['#FFFFFF']
    eigenvalues = []
    x_pad = 0.2
    y_pad = 0.2

    for key, value in kwargs.items():
        if key == 'figure_size':
            figure_size = value
        if key == 'sample_df_x':
            sample_df_x = value
        if key == 'sample_df_y':
            sample_df_y = value
        if key == 'samp_col':
            samp_col = value
        if key == 'samp_alpha':
            samp_alpha = value
        if key == 'samp_marker':
            samp_marker = value
        if key == 'samp_ms':
            samp_ms = value
        if key == 'samp_leg_loc':
            samp_leg_loc = value
        if key == 'feature_df_x':
            sample_df_x = value
        if key == 'feature_df_y':
            sample_df_y = value
        if key == 'feat_col':
            feat_col = value
        if key == 'feat_alpha':
            feat_alpha = value
        if key == 'arrow_width':
            arrow_width = value
        if key == 'arrow_head':
            arrow_head = value
        if key == 'taxa_rank':
            taxa_rank = value
        if key == 'feat_leg_loc':
            feat_leg_loc = value
        if key == 'feature_order':
            if value == 0:
                feature_order = 0
            if value == 1:
                feature_order = 1
        if key == 'sample_drop_list':
            sample_drop_list = value
        if key == 'show_color_drop':
            show_color_drop = value
        if key == 'sample_drop_col':
            sample_drop_col = value
        if key == 'eigenvalues':
            eigenvalues = value
        if key == 'x_pad':
            x_pad = value
        if key == 'y_pad':
            y_pad = value

    if not isinstance(sample_df, pd.core.frame.DataFrame):
        raise ValueError('`sample_df` must be a `pd.DataFrame`, '
                         'not %r.' % type(sample_df).__name__)

    if feature_df is not None:
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

    if sample_color_dict is not None:
        if not isinstance(sample_color_dict, dict):
            raise ValueError('`sample_color_dict` must be a `dictionary`, '
                             'not %r.' % type(sample_color_dict).__name__)

    if feature_color_dict is not None:
        if not isinstance(feature_color_dict, dict):
            raise ValueError('`feature_color_dict` must be a `dictionary`, '
                             'not %r.' % type(feature_color_dict).__name__)

    if sample_df_metadata is not None and sample_color_dict is None:
        if sample_color_category is None:
            raise ValueError('sample_color_category must be a specified')

    if sample_df_metadata is not None and sample_color_dict is not None:
        if sample_color_category is None:
            raise ValueError('sample_color_category must be a specified')

    if feature_df_metadata is not None and feature_color_dict is not None:
        if feature_color_category is None:
            raise ValueError('feature_color_category must be a specified')

    if sample_drop_list is not None:
        if not isinstance(sample_drop_list, list):
            raise ValueError('`sample_drop_list` must be a `list`, '
                             'not %r.' % type(sample_drop_list).__name__)

    if sample_drop_col is not None:
        if not isinstance(sample_drop_col, list):
            raise ValueError('`sample_drop_col` must be a `list`, '
                             'not %r.' % type(sample_drop_col).__name__)

    if sample_df_metadata is not None:
        if (sample_df.index != sample_df_metadata.index).any():
            sample_df = sample_df.sort_index(axis=0)
            sample_df_metadata = sample_df_metadata.sort_index(axis=0)

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)

    sample_colors = cm = plt.get_cmap(samp_col)
    feature_colors = cm = plt.get_cmap(feat_col)

    sample_group_append = []
    colorVal = []

    if sample_df_metadata is None:
        print("Sample Block 1")
        ax.plot(np.ravel(sample_df[sample_df_x]),
                np.ravel(sample_df[sample_df_x]),
                marker=samp_marker, linestyle='',
                ms=samp_ms, alpha=samp_alpha)

    if sample_df_metadata is not None and sample_color_dict is None:
        print("Sample Block 2")
        sample_df_copy = sample_df
        sample_df_metadata_copy = sample_df_metadata

        if len(sample_drop_list) > 0:
            index_drop = sample_df_metadata[sample_color_category]
            .apply(lambda x: x not in sample_drop_list)
            sample_df_metadata = sample_df_metadata.loc[index_drop]
            sample_df = sample_df.loc[sample_df_metadata.index]

            for name, group in
            sample_df.groupby(sample_df_metadata[sample_color_category]):
                sample_group_append.append(name)
                sample_group_append = sorted(list(set(sample_group_append)))
                cNorm = colors.Normalize(vmin=0,
                                         vmax=(len(sample_group_append)-1))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sample_colors)

            for index, row in enumerate(sample_group_append):
                colorVal.append(scalarMap.to_rgba(index))

            if show_color_drop is False:
                sample_color_dict = dict(zip(sample_group_append, colorVal))
                sample_color_dict = OrderedDict(sorted(sample_color_dict
                                                       .items(),
                                                       key=lambda x: x[0],
                                                       reverse=True))

                for name, group in
                sample_df.groupby(sample_df_metadata[sample_color_category]):
                    ax.plot(np.ravel(group[sample_df_x]),
                            np.ravel(group[sample_df_y]),
                            marker=samp_marker, linestyle='', ms=samp_ms,
                            color=sample_color_dict[name],
                            label=name, alpha=samp_alpha)

            if show_color_drop is True:

                color_drop_append = []
                if len(sample_drop_col) == 1:
                    for index in range(len(sample_drop_list)):
                        color_drop_append.append(sample_drop_col[0])
                    colorVal = colorVal + color_drop_append

                if len(sample_drop_col) == len(sample_drop_list):
                    for index in range(len(sample_drop_list)):
                        color_drop_append.append(sample_drop_col[index])
                    colorVal = colorVal + color_drop_append

                sample_group_append = list(sample_group_append) +
                list(sample_drop_list)
                sample_color_dict = dict(zip(sample_group_append, colorVal))
                sample_color_dict = OrderedDict(sorted(sample_color_dict
                                                       .items(),
                                                       key=lambda x: x[0],
                                                       reverse=True))

                for name, group in sample_df_copy
                .groupby(sample_df_metadata_copy[sample_color_category]):
                    if name not in sample_drop_list:
                        ax.plot(np.ravel(group[sample_df_x]),
                                np.ravel(group[sample_df_y]),
                                marker=samp_marker, linestyle='', ms=samp_ms,
                                color=sample_color_dict[name],
                                label=name, alpha=samp_alpha)

                for name, group in sample_df_copy
                .groupby(sample_df_metadata_copy[sample_color_category]):
                    if name in sample_drop_list:
                        ax.plot(np.ravel(group[sample_df_x]),
                                np.ravel(group[sample_df_y]),
                                marker=samp_marker, linestyle='', ms=samp_ms,
                                color=sample_color_dict[name],
                                label=name, alpha=samp_alpha)

        if not sample_drop_list:

            sample_group_append = []
            for name, group in
            sample_df.groupby(sample_df_metadata[sample_color_category]):
                sample_group_append.append(name)
                sample_group_append = sorted(list(set(sample_group_append)))
                cNorm = colors.Normalize(vmin=0,
                                         vmax=(len(sample_group_append)-1))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sample_colors)

            for index, row in enumerate(sample_group_append):
                colorVal.append(scalarMap.to_rgba(index))

            sample_color_dict = dict(zip(sample_group_append, colorVal))
            sample_color_dict = OrderedDict(sorted(sample_color_dict.items(),
                                                   key=lambda x: x[0],
                                                   reverse=True))

            for name, group in
            sample_df.groupby(sample_df_metadata[sample_color_category]):
                ax.plot(np.ravel(group[sample_df_x]),
                        np.ravel(group[sample_df_y]),
                        marker=samp_marker, linestyle='', ms=samp_ms,
                        color=sample_color_dict[name],
                        label=name, alpha=samp_alpha)

        sample_color_dict = None

    if sample_df_metadata is not None and sample_color_dict is not None:
        print("Sample Block 3")
        if len(sample_drop_list) > 0:
            index_drop = sample_df_metadata[sample_color_category]
            .apply(lambda x: x not in sample_drop_list)
            sample_df_metadata = sample_df_metadata.loc[index_drop]
            sample_df = sample_df.loc[sample_df_metadata.index]

            sample_color_dict = OrderedDict(sorted(sample_color_dict.items(),
                                                   key=lambda x: x[0],
                                                   reverse=True))

            for name, group in
            sample_df.groupby(sample_df_metadata[sample_color_category]):
                ax.plot(np.ravel(group[sample_df_x]),
                        np.ravel(group[sample_df_y]),
                        marker=samp_marker, linestyle='', ms=samp_ms,
                        color=sample_color_dict[name],
                        label=name, alpha=samp_alpha)

        if not sample_drop_list:
            sample_color_dict = OrderedDict(sorted(sample_color_dict.items(),
                                                   key=lambda x: x[0],
                                                   reverse=True))
            for name, group in
            sample_df.groupby(sample_df_metadata[sample_color_category]):
                ax.plot(np.ravel(group[sample_df_x]),
                        np.ravel(group[sample_df_y]),
                        marker=samp_marker, linestyle='', ms=samp_ms,
                        color=sample_color_dict[name],
                        label=name, alpha=samp_alpha)

        sample_color_dict = None

    ax2 = ax.twinx()

    if sample_color_category is not None:
        ax.legend(title=sample_color_category, loc=samp_leg_loc, numpoints=1)
    else:
        ax.legend(loc=samp_leg_loc, numpoints=1)

    ax2.set_ylim(ax.get_ylim())

    recs = []
    feature = []
    otu_feature_color = []
    otu_feature_append = []
    colorVal = []

    if feature_df is not None and feature_df_metadata is None:
        print('Feature Block 1')
        if feature_color_category is None:
            feature_color_category = 'Taxonomy'
        for name in feature_df.index:
            otu_feature = name.split(';')[taxa_rank].replace(' ', '')
            otu_feature_append.append(otu_feature)

        otu_feature_append = sorted(list(set(otu_feature_append)))
        cNorm = colors.Normalize(vmin=0, vmax=(len(otu_feature_append)-1))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=feature_colors)

        for index, row in enumerate(otu_feature_append):
            colorVal.append(scalarMap.to_rgba(index))

        feature_color_dict = dict(zip(otu_feature_append, colorVal))
        feature_color_dict = OrderedDict(sorted(feature_color_dict.items(),
                                                key=lambda x: x[0]))

        otu_feature_append = []
        for name, row in feature_df.iterrows():
            otu_feature = name.split(';')[taxa_rank].replace(' ', '')
            ax2.arrow(0, 0,
                      np.asscalar(row[feature_df_x]),
                      np.asscalar(row[feature_df_y]),
                      width=arrow_width, head_width=arrow_head,
                      alpha=feat_alpha,
                      color=feature_color_dict[otu_feature])

        for key, value in feature_color_dict.items():
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=feature_color_dict[key],
                                           alpha=feat_alpha))
            feature.append(key)

        ax2.legend(recs, feature, loc=feat_leg_loc,
                   title=feature_color_category)

    if feature_df_metadata is not None and feature_color_dict is None:
        print('Feature Block 2')
        if feature_color_category is None:
            feature_color_category = 'Taxonomy'

        for name, group in
        feature_df.groupby(feature_df_metadata[feature_color_category]):
            if len(name.split(';')) > 1:
                otu_feature = name.split(';')[taxa_rank].replace(' ', '')
                otu_feature_append.append(otu_feature)
            else:
                otu_feature = name
                otu_feature_append.append(otu_feature)

        otu_feature_append = sorted(list(set(otu_feature_append)))
        cNorm = colors.Normalize(vmin=0, vmax=(len(otu_feature_append)-1))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=feature_colors)

        for index, row in enumerate(otu_feature_append):
            colorVal.append(scalarMap.to_rgba(index))
        feature_color_dict = dict(zip(otu_feature_append, colorVal))
        feature_color_dict = OrderedDict(sorted(feature_color_dict.items(),
                                                key=lambda x: x[0]))

        otu_feature_append = []
        for name, group in
        feature_df.groupby(feature_df_metadata[feature_color_category]):
            if len(name.split(';')) > 1:
                otu_feature = name.split(';')[taxa_rank].replace(' ', '')
                otu_feature_append.append(otu_feature)
                ax2.arrow(0, 0,
                          np.asscalar(group[feature_df_x]),
                          np.asscalar(group[feature_df_y]),
                          width=arrow_width, head_width=arrow_head,
                          alpha=feat_alpha,
                          color=feature_color_dict[otu_feature])
            else:
                otu_feature = name
                otu_feature_append.append(otu_feature)
                ax2.arrow(0, 0,
                          np.asscalar(group[feature_df_x]),
                          np.asscalar(group[feature_df_x]),
                          width=arrow_width, head_width=arrow_head,
                          alpha=feat_alpha,
                          color=feature_color_dict[otu_feature])

        for key, value in feature_color_dict.items():
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=feature_color_dict[key],
                                           alpha=feat_alpha))
            feature.append(key)
        ax2.legend(recs, feature, loc=feat_leg_loc,
                   title=feature_color_category)
        feature_color_dict = None

    if feature_df_metadata is not None and
    feature_color_dict is not None and
    feature_color_category is not None:
        print('Feature Block 3')
        for name, group in
        feature_df.groupby(feature_df_metadata[feature_color_category]):
            if len(name.split(';')) > 1:
                otu_feature = name.split(';')[taxa_rank].replace(' ', '')
                otu_feature_append.append(otu_feature)
            else:
                otu_feature = name
                otu_feature_color.append(feature_color_dict[otu_feature])
                otu_feature_append.append(otu_feature)

        for name, group in
        feature_df.groupby(feature_df_metadata[feature_color_category]):
            if len(name.split(';')) > 1:
                otu_feature = name.split(';')[taxa_rank].replace(' ', '')
                otu_feature_append.append(otu_feature)
                ax2.arrow(0, 0,
                          np.asscalar(group[feature_df_x]),
                          np.asscalar(group[feature_df_y]),
                          width=arrow_width, head_width=arrow_head,
                          alpha=feat_alpha,
                          color=feature_color_dict[otu_feature])
            else:
                otu_feature = name
                otu_feature_append.append(otu_feature)
                ax2.arrow(0, 0,
                          np.asscalar(group[feature_df_x]),
                          np.asscalar(group[feature_df_y]),
                          width=arrow_width, head_width=arrow_head,
                          alpha=feat_alpha,
                          color=feature_color_dict[otu_feature])

        for key, value in feature_color_dict.items():
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=feature_color_dict[key],
                                           alpha=feat_alpha))
            feature.append(key)

        ax2.legend(
                    recs, feature, loc=feat_leg_loc,
                    title=feature_color_category)

    if feature_df is not None:
        xmin = min([min(sample_df.ix[:, 0]), min(feature_df.ix[:, 0])])
        xmax = max([max(sample_df.ix[:, 0]), max(feature_df.ix[:, 0])])
        ymin = min([min(sample_df.ix[:, 1]), min(feature_df.ix[:, 1])])
        ymax = max([max(sample_df.ix[:, 1]), max(feature_df.ix[:, 1])])
        xpad = (xmax - xmin) * x_pad
        ypad = (ymax - ymin) * y_pad

        ax.set_zorder(ax2.get_zorder()+(1-feature_order))
        ax.patch.set_visible(False)

        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, xmax + ypad)
        ax2.set_xlim(xmin - xpad, xmax + xpad)
        ax2.set_ylim(ymin - ypad, xmax + ypad)
        ax2.set_yticks([])
    else:
        xmin = min([min(sample_df.ix[:, 0])])
        xmax = max([max(sample_df.ix[:, 0])])
        ymin = min([min(sample_df.ix[:, 1])])
        ymax = max([max(sample_df.ix[:, 1])])
        xpad = (xmax - xmin) * x_pad
        ypad = (ymax - ymin) * y_pad

        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, xmax + ypad)
        ax2.set_yticks([])

    if len(eigenvalues) > 0:
        ax.set_xlabel(
                      'PC 1 ({:.2%})'
                      .format(eigenvalues[0]**2/sum(eigenvalues**2)))
        ax.set_ylabel(
                      'PC 2 ({:.2%})'
                      .format(eigenvalues[1]**2/sum(eigenvalues**2)))

    return fig, [ax, ax2]
