from .utils import *
import matplotlib.pyplot as plt
from gpgraph.draw import *
from scipy import sparse
import numpy as np
from numpy import inf
import matplotlib as mpl


def plot_timescales(timescales, figsize=None, n=None, color='orange'):
    """Simple bar plot of a sequence of values"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i for i in range(0, len(timescales[:n]))], timescales[:n], color=color)
    ax.set_title("Timescales")
    return fig, ax

def plot_eigenvalues(eigenvalues, figsize=None, n=None, color='orange'):
    """Simple bar plot of a sequence of values"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i for i in range(0, len(eigenvalues[:n]))], eigenvalues[:n], color=color)
    ax.set_title("Eigenvalues")
    return fig, ax


def plot_matrix(matrix, log=True, remove_diag=False, colorbar=True, figsize=(5, 5), ax=None, scale_x=True):
    """Plot the entries of a matrix

    Parameters
    ----------
    matrix : 2D numpy.ndarray.
        A matrix with numerical entries.

    log : bool (default=True).
        log10 transform data to visualize low-valued matrix entries (recommended for transition matrices).

    remove_diag : bool (default=False).
        Remove diagonal if True. Helps with visualizing matrix where diagonal is very dominant.

    colorbar : bool (default=True)
        If True plot colorbar mapping values to color.

    figsize : tuple of int (default=(12,10).
        Size of matplotlib figure.

    scale_x : bool (default=True)
        If True, markers will be scaled to fill one xaxis unit, if False markers will be scaled to fill one yaxis unit.
        Both at the same time is not possible.

    Returns
    -------
    fig : matplotlib figure.
        matplotlib figure of size 'figsize'.

    ax : matplotlib axis.
        matplotlib axis that contains the matrix visualization.

    """
    T = matrix.copy()

    if remove_diag:
        np.fill_diagonal(T, val=0)

    if log:
        T = np.log10(T)
        # Values that are too small will become -inf, we set them to the min. value of T
        T[T == -inf] = 1
        minv = np.min(T)
        T[T == 1] = minv

    # Normalize colors to min and max of T.
    norm = mpl.colors.Normalize(vmin=np.min(T), vmax=np.max(T))

    cmap = mpl.cm.get_cmap("Greys")

    # Get a color for each value.
    colors = [cmap(norm(val)) for val in list(T.flatten())]

    # Get coordinates.
    indices = np.indices(T.shape)
    x = indices[1].flatten()
    y = indices[0].flatten()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if colorbar:
        # Get color map
        cm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cm.set_array([])

        fig.colorbar(cm)

    ax.set_xlim(-1, T.shape[1])
    ax.set_ylim(-1, T.shape[0])
    ax.invert_yaxis()

    # Get number of pixels per axis unit and calculate scatter marker size that fills exactly one axis unit.
    x_pix, y_pix = ax.transData.transform([1, 1]) - ax.transData.transform((0, 0))
    if scale_x:
        s = x_pix ** 2  # Diameter of a marker in pixels is equal to the square root of markersize s.

    else:
        s = y_pix ** 2

    ax.scatter(x, y, c=colors, cmap='Greys', s=s, marker='s')

    return fig, ax


def plot_network(
    network,
    flux=None,
    ax=None,
    figsize=(15,10),
    nodelist=[],
    attribute="phenotypes",
    vmin=None,
    vmax=None,
    cmap="YlOrRd",
    cmap_truncate=False,
    colorbar=False,
    node_labels="genotypes",
    edge_scalar=15.0,
    edge_labels=False,
    edge_color='k',
    style='solid',
    edge_alpha=1.0,
    arrows=False,
    arrowstyle='-|>',
    arrowsize=10,
    node_size=3000,
    node_color='r',
    node_shape='o',
    alpha=1.0,
    linewidths=0,
    edgecolors="black",
    label=None,
    **kwds):
    """Draw the GenotypePhenotypeGraph using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See :py:mod:`networkx.drawing.layout` for functions that
       compute node positions.

    arrows : bool, optional (default=False)
       For directed graphs, if True draw arrowheads.

    with_labels :  bool, optional (default=True)
       Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional (default G.nodes())
       Draw only specified nodes

    attribute : string (default = "phenotypes")
       node attribute that is used to set the color

    edgelist : list, optional (default=G.edges())
       Draw only specified edges

    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    node_color : color string, or array of floats, (default=phenotypes)
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_shape :  string, optional (default='o')
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8'.

    alpha : float, optional (default=1.0)
       The node and edge transparency

    cmap : Matplotlib colormap, optional (default='plasmas')
       Colormap for mapping intensities of nodes

    vmin,vmax : float, optional (default=None)
       Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    width : float, optional (default=1.0)
       Line width of edges

    color_bar : False
        If True, show colorbar for nodes.

    edge_color : color string, or array of floats (default='gray')
       Edge color. Can be a single color format string,
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional (default=None)
       Minimum and maximum for edge colormap scaling

    style : string, optional (default='solid')
       Edge line style (solid|dashed|dotted,dashdot)

    labels : dictionary, optional (default='genotypes')
       Node labels in a dictionary keyed by node of text labels

    font_size : int, optional (default=12)
       Font size for text labels

    font_color : string, optional (default='k' black)
       Font color string

    font_weight : string, optional (default='normal')
       Font weight

    font_family : string, optional (default='sans-serif')
       Font family

    label : string, optional
        Label for graph legend

    Notes
    -----
    For directed graphs, "arrows" (actually just thicker stubs) are drawn
    at the head end.  Arrows can be turned off with keyword arrows=False.
    Yes, it is ugly but drawing proper arrows with Matplotlib this
    way is tricky.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Flattened position.
    pos = flattened(network, vertical=True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if flux is None:
        # All edges
        edgelist = network.edges()
        width = 1
    else:
        # Get flux through edges
        edgelist = flux.keys()
        width = [edge_scalar * flx for flx in flux.values()]

    if not nodelist:
        nodelist = list(network.nodes().keys())

    if vmax is None:
        attributes = list(nx.get_node_attributes(network, name=attribute).values())
        vmin = min(attributes)
        vmax = max(attributes)

    if cmap_truncate:
        cmap = truncate_colormap(cmap, minval=0.05, maxval=0.95)

    if node_labels:
        labels = {n: network.nodes[n][node_labels] for n in nodelist}
    else:
        labels = None

    # Default options
    node_options = dict(
        nodelist=nodelist,
        vmin=vmin,
        vmax=vmax,
        node_shape=node_shape,
        node_size=node_size,
        node_color=[network.nodes[n][attribute] for n in nodelist],
        linewidths=linewidths,
        edgecolors=edgecolors,
        cmap=cmap,
        labels=labels,
        cmap_truncate=False,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G=network,
        pos=pos,
        edgelist=edgelist,
        width=width,
        edge_color=edge_color,
        ax=ax,
        style=style,
        alpha=edge_alpha,
        arrows=arrows,
        arrowstyle=arrowstyle,
        arrowsize=arrowsize,
    )

    # Draw nodes.
    nx.draw_networkx_nodes(
        G=network,
        pos=pos,
        ax=ax,
        **node_options
    )

    # Draw labels again manually because I can't find the bug that causes networkx to ignore labels.
    if labels:
        nx.draw_networkx_labels(network, pos=pos, labels=labels)

    if flux is not None:
        edge_flux = {edge: round(flux[edge],2) for edge in edgelist}
        # Draw edge labels
        if edge_labels:
            nx.draw_networkx_edge_labels(network, pos=pos, edge_labels=edge_flux,
                                     label_pos=0.5, font_size=10, font_color='k',
                                     font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None,
                                     rotate=True, **kwds)

    # Add a colorbar?
    if colorbar:
        norm = mpl.colors.Normalize(
            vmin=vmin,
            vmax=vmax)

        # create a ScalarMappable and initialize a data structure
        cm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cm.set_array([])
        fig.colorbar(cm)

    return fig, ax


def plot_clusters(
    network,
    clusters,
    xaxis='forward_committor',
    yaxis='fitness',
    flux=None,
    ax=None,
    figsize=(15,10),
    nodelist=[],
    cluster_scale=0.1,
    attribute="phenotypes",
    vmin=None,
    vmax=None,
    cmap="YlOrRd",
    cmap_truncate=False,
    colorbar=False,
    node_labels="genotypes",
    edge_scalar=15.0,
    edge_color='k',
    edge_labels=False,
    style='solid',
    edge_alpha=1.0,
    arrows=False,
    arrowstyle='-|>',
    arrowsize=10,
    node_size=3000,
    node_color='r',
    node_shape='o',
    alpha=1.0,
    linewidths=0,
    edgecolors="black",
    label=None,
    **kwds):
    """Draw the GenotypePhenotypeGraph using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See :py:mod:`networkx.drawing.layout` for functions that
       compute node positions.

    arrows : bool, optional (default=False)
       For directed graphs, if True draw arrowheads.

    with_labels :  bool, optional (default=True)
       Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional (default G.nodes())
       Draw only specified nodes

    attribute : string (default = "phenotypes")
       node attribute that is used to set the color

    edgelist : list, optional (default=G.edges())
       Draw only specified edges

    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    node_color : color string, or array of floats, (default=phenotypes)
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_shape :  string, optional (default='o')
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8'.

    alpha : float, optional (default=1.0)
       The node and edge transparency

    cmap : Matplotlib colormap, optional (default='plasmas')
       Colormap for mapping intensities of nodes

    vmin,vmax : float, optional (default=None)
       Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    width : float, optional (default=1.0)
       Line width of edges

    color_bar : False
        If True, show colorbar for nodes.

    edge_color : color string, or array of floats (default='gray')
       Edge color. Can be a single color format string,
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional (default=None)
       Minimum and maximum for edge colormap scaling

    style : string, optional (default='solid')
       Edge line style (solid|dashed|dotted,dashdot)

    labels : dictionary, optional (default='genotypes')
       Node labels in a dictionary keyed by node of text labels

    font_size : int, optional (default=12)
       Font size for text labels

    font_color : string, optional (default='k' black)
       Font color string

    font_weight : string, optional (default='normal')
       Font weight

    font_family : string, optional (default='sans-serif')
       Font family

    label : string, optional
        Label for graph legend

    Notes
    -----
    For directed graphs, "arrows" (actually just thicker stubs) are drawn
    at the head end.  Arrows can be turned off with keyword arrows=False.
    Yes, it is ugly but drawing proper arrows with Matplotlib this
    way is tricky.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Positions of circular clusters.
    pos = cluster_positions(network, clusters, xaxis=xaxis, yaxis=yaxis, scale=cluster_scale)
    ax.set_ylim(0.92, 1.05)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.autoscale(enable=True)
    ax.set_xlabel("Forward Committor", size=15)
    ax.set_ylabel("Fitness", size=15)
    ax.axis("auto")

    if flux is None:
        # All edges
        edgelist = network.edges()
        width = edge_scalar * 1
    else:
        # Get flux through edges
        peaks = cluster_peaks(network, clusters)
        indices = np.nonzero(flux)
        cluster_edges = list(zip(indices[0], indices[1]))
        # translate the edges between cluster to edges between the peaks of each cluster.
        edgelist = [(peaks[edge[0]], peaks[edge[1]]) for edge in cluster_edges]  # Map cluster index to peak node index.
        width = edge_scalar * flux[flux > 0]

    if not nodelist:
        nodelist = list(network.nodes().keys())

    if vmax is None:
        attributes = list(nx.get_node_attributes(network, name=attribute).values())
        vmin = min(attributes)
        vmax = max(attributes)

    if cmap_truncate:
        cmap = truncate_colormap(cmap, minval=0.05, maxval=0.95)

    if node_labels:
        labels = {n: network.nodes[n][node_labels] for n in nodelist}
    else:
        labels = None

    # Default options
    node_options = dict(
        nodelist=nodelist,
        vmin=vmin,
        vmax=vmax,
        node_shape=node_shape,
        node_size=node_size,
        node_color=[network.nodes[n][attribute] for n in nodelist],
        linewidths=linewidths,
        edgecolors=edgecolors,
        cmap=cmap,
        labels=labels,
        cmap_truncate=False,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G=network,
        pos=pos,
        edgelist=edgelist,
        width=width,
        edge_color=edge_color,
        ax=ax,
        style=style,
        alpha=edge_alpha,
        arrows=arrows,
        arrowstyle=arrowstyle,
        arrowsize=arrowsize,
    )

    # Draw nodes.
    nx.draw_networkx_nodes(
        G=network,
        pos=pos,
        ax=ax,
        **node_options
    )

    # Draw labels again manually because I can't find the bug that causes networkx to ignore labels.
    if labels:
        nx.draw_networkx_labels(network, pos=pos, labels=labels)

    if flux is not None:
        rounded = np.round(flux, decimals=2)
        edgeflux = sparse.dok_matrix(rounded)
        # Draw edge labels
        if edge_labels:
            nx.draw_networkx_edge_labels(network, pos=pos, edge_labels=edgeflux,
                                     label_pos=0.5, font_size=10, font_color='k',
                                     font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None,
                                     rotate=True, **kwds)

    # Add a colorbar?
    if colorbar:
        norm = mpl.colors.Normalize(
            vmin=vmin,
            vmax=vmax)

        # create a ScalarMappable and initialize a data structure
        cm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cm.set_array([])
        fig.colorbar(cm)

    return fig, ax



# def plot_clusters(network, clusters, scale=1, figsize=(10,10)):
#     print(clusters)
#     pos = cluster_positions(network, clusters, scale=scale)
#     print(pos)
#
#     #fig, ax = plt.subplots(figsize=figsize)
#     fig, ax = draw_flattened(network, pos=pos)
#
#
#     ax.spines['left'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)
#     ax.set_xticks([i for i in np.arange(0, 1.1, 0.1)])
#     ax.set_yticks([i for i in np.arange(0, 1.1, 0.05)])
#     ax.autoscale(enable=True)
#     ax.set_xlabel("Forward Committor", size=15)
#     ax.set_ylabel("Fitness", size=15)
#
#     return fig, ax