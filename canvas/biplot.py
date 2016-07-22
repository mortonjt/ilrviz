from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
from collections import OrderedDict


def make_biplot(samples,
                features=None,
                sample_metadata=None,
                feature_metadata=None,
                sample_color=None,
                feature_color=None,
                **kwargs):

    """ Creates a biplot.

    Creates a 2D biplot where the points correspond to samples and the
    vectors correspond to features.  The distance between points often
    correspond to sample dissimilarity and the angle between vectors
    is often a proxy for correlations between vectors.

    Parameters
    ----------
    samples : pd.DataFrame
        A table of samples by principal coordinates.
    features: pd.DataFrame
       A table of features by principal coordinates.
    sample_metadata: pd.Series
       An optional series for specifying metadata for samples
       to use in plotting the sample points.
    feature_metadata: pd.Series
       An optional series for specifying metadata for features
       to use in plotting the feature vectors.
    sample_color: dict, or OrderedDict, or pd.Series
      Dictionary, or Ordereded Dictionary, or pd.Series of
      colors corresponding to different samples, or
      categories in `sample_metadata`.
    feature_color: dict, or OrderedDict, or pd.Series
      Dictionary, or Ordereded Dictionary, or pd.Series of
      colors corresponding to different features, or
      categories in `sample_metadata`.

    Returns
    -------
    matplotlib.figure.Figure
        Container for biplot figure
    list of matplotlib.axes._subplots.AxesSubplot
       Two matplotlib axes objects.  The first axes object contains
       plotting information about the samples.  The second axes
       object contains plotting information about the features.
    """

    figure_size = (15, 15)
    samples_x = 'PCA1'
    samples_y = 'PCA2'
    samp_col_set = 'RdGy'
    samp_alpha = 1
    samp_marker = 'o'
    samp_ms = 8
    samp_leg_loc = 2
    features_x = 'PCA1'
    features_y = 'PCA2'
    feat_col_set = 'Set1'
    feat_alpha = 1
    arrow_width = 0.02
    arrow_head = 0.05
    feat_leg_loc = 1
    feat_leg_ncol = 1
    feat_leg_ls = 0.5
    feature_order = 0
    samp_drop_list = []
    show_color_drop = False
    samp_drop_col = ['#FFFFFF']
    eigenvalues = []
    x_pad = 0.3
    y_pad = 0.3
    show_feat_legend = True

    for key, value in kwargs.items():
        if key == 'figure_size':
            figure_size = value
        if key == 'samples_x':
            samples_x = value
        if key == 'samples_y':
            samples_y = value
        if key == 'samp_col_set':
            samp_col_set = value
        if key == 'samp_alpha':
            samp_alpha = value
        if key == 'samp_marker':
            samp_marker = value
        if key == 'samp_ms':
            samp_ms = value
        if key == 'samp_leg_loc':
            samp_leg_loc = value
        if key == 'features_x':
            samples_x = value
        if key == 'features_y':
            samples_y = value
        if key == 'feat_col_set':
            feat_col_set = value
        if key == 'feat_alpha':
            feat_alpha = value
        if key == 'arrow_width':
            arrow_width = value
        if key == 'arrow_head':
            arrow_head = value
        if key == 'feat_leg_loc':
            feat_leg_loc = value
        if key == 'feat_leg_ncol':
            feat_leg_ncol = value
        if key == 'feat_leg_ls':
            feat_leg_ls = value
        if key == 'feature_order':
            if value == 0:
                feature_order = 0
            if value == 1:
                feature_order = 1
        if key == 'samp_drop_list':
            samp_drop_list = value
        if key == 'show_color_drop':
            show_color_drop = value
        if key == 'samp_drop_col':
            samp_drop_col = value
        if key == 'eigenvalues':
            eigenvalues = value
        if key == 'x_pad':
            x_pad = value
        if key == 'y_pad':
            y_pad = value
        if key == 'show_feat_legend':
            show_feat_legend = value

    if not isinstance(samples, pd.core.frame.DataFrame):
        raise ValueError('`samples` must be a `pd.DataFrame`, '
                         'not %r.' % type(samples).__name__)

    if features is not None:
        if not isinstance(features, pd.core.frame.DataFrame):
            raise ValueError('`features` must be a `pd.DataFrame`, '
                             'not %r.' % type(features).__name__)

    if sample_metadata is not None:
        if not isinstance(sample_metadata, pd.core.series.Series):
            raise ValueError('`sample_metadata` must be a `pd.Series`, '
                             'not %r.' % type(sample_metadata).__name__)

    if feature_metadata is not None:
        if not isinstance(feature_metadata, pd.core.series.Series):
            raise ValueError('`feature_metadata` must be a `pd.Series`, '
                             'not %r.' % type(feature_metadata).__name__)
        if features is None:
            raise ValueError('`features` is missing')

    if sample_color is not None:
        if (not isinstance(sample_color, dict) and not
                isinstance(sample_color, pd.core.series.Series) and not
                isinstance(sample_color, OrderedDict)):
            raise ValueError('`sample_color` must be a `dictionary` or '
                             '`OrderedDict` or `pd.Series`, '
                             'not %r.' % type(sample_color).__name__)

    if feature_color is not None:
        if (not isinstance(feature_color, dict) and not
                isinstance(feature_color, pd.core.series.Series) and not
                isinstance(feature_color, OrderedDict)):
            raise ValueError('`sample_color` must be a `dictionary` or '
                             '`OrderedDict` or `pd.Series`, '
                             'not %r.' % type(feature_color).__name__)

    if samp_drop_list is not None:
        if not isinstance(samp_drop_list, list):
            raise ValueError('`samp_drop_list` must be a `list`, '
                             'not %r.' % type(samp_drop_list).__name__)

    if samp_drop_col is not None:
        if not isinstance(samp_drop_col, list):
            raise ValueError('`samp_drop_col` must be a `list`, '
                             'not %r.' % type(samp_drop_col).__name__)

    if sample_metadata is not None:
        if (samples.index != sample_metadata.index).any():
            samples = samples.sort_index(axis=0)
            sample_metadata = sample_metadata.sort_index(axis=0)
        if not (sample_metadata.index.equals(samples.index)):
                raise ValueError('`samples` and `sample_metadata` '
                                 'do not have matching indices')

    if feature_metadata is not None:
        if (features.index != feature_metadata.index).any():
            features = features.sort_index(axis=0)
            feature_metadata = feature_metadata.sort_index(axis=0)
        if not (feature_metadata.index.equals(features.index)):
                raise ValueError('`features` and `feature_metadata` '
                                 'do not have matching indices')

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)

    sample_colors = plt.get_cmap(samp_col_set)
    feature_colors = plt.get_cmap(feat_col_set)

    sample_group_append = []
    colorVal = []

    if sample_metadata is None:
        ax.plot(np.ravel(samples[samples_x]),
                np.ravel(samples[samples_y]),
                marker=samp_marker, linestyle='',
                ms=samp_ms, alpha=samp_alpha)

    if sample_metadata is not None:

        if sample_color is not None:
            if (type(sample_color) is dict or
                    type(sample_color) is OrderedDict):
                sample_color = pd.Series(sample_color.values(),
                                         index=sample_color.keys())

        if len(samp_drop_list) > 0:

            def dropf(x):
                return x not in samp_drop_list
            index_drop = sample_metadata.apply(dropf)

            def keepf(x):
                return x in samp_drop_list
            index_keep = sample_metadata.apply(keepf)

            samp_meta_b = sample_metadata.loc[index_drop]
            samp_b = samples.loc[samp_meta_b.index]

            samp_meta_s = sample_metadata.loc[index_keep]
            samp_s = samples.loc[samp_meta_s.index]

            if sample_color is None:
                for name, group in samp_b.groupby(samp_meta_b):
                    sample_group_append.append(name)
                sample_group_append = sorted(list(set(sample_group_append)))
                cNorm = colors.Normalize(vmin=0,
                                         vmax=(len(sample_group_append)-1))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sample_colors)

                for index, row in enumerate(sample_group_append):
                    colorVal.append(scalarMap.to_rgba(index))

                sample_color = pd.Series(colorVal, index=sample_group_append)
            else:
                sample_color = sample_color.drop(samp_drop_list)

            samp_meta_b = samp_meta_b.astype("category",
                                             categories=sample_color.keys(),
                                             ordered=True)
            samp_meta_b = samp_meta_b.dropna().sort_values()

            for name, group in samp_b.groupby(samp_meta_b):
                ax.plot(np.ravel(group[samples_x]),
                        np.ravel(group[samples_y]),
                        marker=samp_marker, linestyle='', ms=samp_ms,
                        color=sample_color[name],
                        label=name, alpha=samp_alpha)

            if show_color_drop:
                colorVal_drop = []
                if len(samp_drop_col) == 1:
                    for index in range(len(samp_drop_list)):
                        colorVal_drop.append(samp_drop_col[0])

                if len(samp_drop_col) == len(samp_drop_list):
                    for index in range(len(samp_drop_list)):
                        colorVal_drop.append(samp_drop_col[index])

                samp_col_drop = pd.Series(colorVal_drop, index=samp_drop_list)

                samp_meta_s = samp_meta_s.astype(
                                            "category",
                                            categories=samp_col_drop.keys(),
                                            ordered=True)
                samp_meta_s = samp_meta_s.dropna().sort_values()

                for name, group in samp_s.groupby(samp_meta_s):
                    ax.plot(np.ravel(group[samples_x]),
                            np.ravel(group[samples_y]),
                            marker=samp_marker, linestyle='', ms=samp_ms,
                            color=samp_col_drop[name],
                            label=name, alpha=samp_alpha)

        else:
            if sample_color is None:
                sample_group_append = []
                for name, group in samples.groupby(sample_metadata):
                    sample_group_append.append(name)

                sample_group_append = sorted(list(set(sample_group_append)))
                cNorm = colors.Normalize(vmin=0,
                                         vmax=(len(sample_group_append)-1))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sample_colors)

                for index, row in enumerate(sample_group_append):
                    colorVal.append(scalarMap.to_rgba(index))

                sample_color = pd.Series(colorVal, index=sample_group_append)

            sample_metadata = sample_metadata.astype(
                                                "category",
                                                categories=sample_color.keys(),
                                                ordered=True)
            sample_metadata.sort_values()

            for name, group in samples.groupby(sample_metadata):
                ax.plot(np.ravel(group[samples_x]),
                        np.ravel(group[samples_y]),
                        marker=samp_marker, linestyle='', ms=samp_ms,
                        color=sample_color[name],
                        label=name, alpha=samp_alpha)

    ax2 = ax.twinx()

    if sample_metadata is not None:
        ax.legend(title=sample_metadata.name, loc=samp_leg_loc, numpoints=1)
    else:
        ax.legend(loc=samp_leg_loc, numpoints=1)

    ax2.set_ylim(ax.get_ylim())

    recs = []
    feature = []
    otu_feature_append = []
    colorVal = []

    if (features is not None and feature_metadata is None):
        for index, row in features.iterrows():
            ax2.arrow(0, 0, row[features_x], row[features_y],
                      width=arrow_width, head_width=arrow_head,
                      alpha=feat_alpha, color='r')

    if (features is not None and feature_metadata is not None):
        feature_groups = features.groupby(feature_metadata)

        if (type(feature_color) is dict or
                type(feature_color) is OrderedDict):
            feature_color = pd.Series(feature_color.values(),
                                      index=feature_color.keys())

        if feature_color is None:
            otu_feature_append = []

            for name, group in feature_groups:
                otu_feature_append.append(name)

            otu_feature_append = sorted(list(set(otu_feature_append)))
            cNorm = colors.Normalize(vmin=0,
                                     vmax=(len(otu_feature_append)-1))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=feature_colors)

            for index, row in enumerate(otu_feature_append):
                colorVal.append(scalarMap.to_rgba(index))

            feature_color = pd.Series(colorVal, index=otu_feature_append)

        for name, group in feature_groups:
            for i in range(group[features_x].shape[0]):
                _id = group.index[i]
                ax2.arrow(0, 0,
                          group.loc[_id, features_x],
                          group.loc[_id, features_y],
                          width=arrow_width, head_width=arrow_head,
                          alpha=feat_alpha,
                          color=feature_color[name])

        if show_feat_legend:
            for key in feature_color.keys():
                recs.append(mpatches.Rectangle((0, 0), 1, 1,
                            fc=feature_color[key],
                            alpha=feat_alpha))
                feature.append(key)
            ax2.legend(recs, feature, loc=feat_leg_loc,
                       title=feature_metadata.name,
                       ncol=feat_leg_ncol, labelspacing=feat_leg_ls)

    if features is not None:
        xmin = min([min(samples.ix[:, 0]), min(features.ix[:, 0])])
        xmax = max([max(samples.ix[:, 0]), max(features.ix[:, 0])])
        ymin = min([min(samples.ix[:, 1]), min(features.ix[:, 1])])
        ymax = max([max(samples.ix[:, 1]), max(features.ix[:, 1])])
        xpad = (xmax - xmin) * x_pad
        ypad = (ymax - ymin) * y_pad

        ax.set_zorder(ax2.get_zorder()+(1-feature_order))
        ax.patch.set_visible(False)

        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax2.set_xlim(xmin - xpad, xmax + xpad)
        ax2.set_ylim(ymin - ypad, ymax + ypad)
        ax2.set_yticks([])
    else:
        xmin = min([min(samples.ix[:, 0])])
        xmax = max([max(samples.ix[:, 0])])
        ymin = min([min(samples.ix[:, 1])])
        ymax = max([max(samples.ix[:, 1])])
        xpad = (xmax - xmin) * x_pad
        ypad = (ymax - ymin) * y_pad

        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax2.set_yticks([])

    if len(eigenvalues) > 2:
        e_0 = eigenvalues[0]
        e_1 = eigenvalues[1]
        ax.set_xlabel('PC 1 ({:.2%})'.format(e_0**2/sum(eigenvalues**2)))
        ax.set_ylabel('PC 2 ({:.2%})'.format(e_1**2/sum(eigenvalues**2)))

    return fig, [ax, ax2]
