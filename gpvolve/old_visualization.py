from .utils import *
import matplotlib.pyplot as plt
from gpgraph.draw import *
from scipy import sparse


def plot_timescales(timescales, figsize=None, n=None, color='orange'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i for i in range(0, len(timescales[:n]))], timescales[:n], color=color)
    ax.set_title("Timescales")
    return fig, ax

def plot_eigenvalues(eigenvalues, figsize=None, n=None, color='orange'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i for i in range(0, len(eigenvalues[:n]))], eigenvalues[:n], color=color)
    ax.set_title("Timescales")
    return fig, ax

def plot_clusters(network, clusters, scale=1, figsize=(10,10)):
    spm = shortest_path_matrix(network)
    pos = cluster_positions(network, clusters, spm, scale=scale)

    #fig, ax = plt.subplots(figsize=figsize)
    fig, ax = draw_flattened(network, pos=pos)


    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks([i for i in np.arange(0, 1.1, 0.1)])
    ax.set_yticks([i for i in np.arange(0, 1.1, 0.05)])
    ax.autoscale(enable=True)
    ax.set_xlabel("Forward Committor", size=15)
    ax.set_ylabel("Fitness", size=15)

    return fig, ax


def draw_network(
    M,
    clusters=None,
    flux=None,
    ax=None,
    figsize=(15,10),
    cluster_scale=1,
    nodelist=[],
    attribute="phenotypes",
    vmin=None,
    vmax=None,
    cmap="YlOrRd",
    cmap_truncate=False,
    colorbar=False,
    labels="genotypes",
    edge_scalar=15.0,
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
        fig = ax.get_figure(figsize=figsize)


    if clusters:
        # Positions of circular clusters.
        pos = cluster_positions(M.network, clusters, scale=cluster_scale)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticks([i for i in np.arange(0, 1.1, 0.1)])
        ax.autoscale(enable=True)
        ax.set_xlabel("Forward Committor", size=15)
        ax.set_ylabel("Fitness", size=15)
        ax.axis("equal")
    else:
        # Flattened position.
        pos = flattened(M.network, vertical=True)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    if flux is None:
        # All edges
        edgelist = M.network.edges()
        width = 1
    elif flux.any() and clusters:
        peaks = cluster_peaks(M.network, clusters)
        centers = cluster_centers(M, peaks)
        indices = np.nonzero(flux)
        edges = list(zip(indices[0], indices[1]))
        edgelist = [(centers[i[0]], centers[i[1]]) for i in edges]
        width = edge_scalar * flux[flux > 0]
    else:
        # Get flux through edges
        indices = np.nonzero(flux)
        edgelist = list(zip(indices[0], indices[1]))
        width = edge_scalar * flux[flux > 0]

    if not nodelist:
        nodelist = list(M.network.nodes().keys())

    if vmax is None:
        attributes = list(nx.get_node_attributes(M.network, name=attribute).values())
        vmin = min(attributes)
        vmax = max(attributes)

    if cmap_truncate:
        cmap = truncate_colormap(cmap, minval=0.05, maxval=0.95)

    # Default options
    node_options = dict(
        nodelist=nodelist,
        vmin=vmin,
        vmax=vmax,
        node_shape=node_shape,
        node_size=node_size,
        node_color=[M.network.nodes[n][attribute] for n in nodelist],
        linewidths=linewidths,
        edgecolors=edgecolors,
        cmap=cmap,
        labels={n: M.network.nodes[n][labels] for n in nodelist},
        cmap_truncate=False,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G=M.network,
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
        G=M.network,
        pos=pos,
        ax=ax,
        **node_options
    )

    # Draw labels again manually because I can't find the bug that causes networkx to ignore labels.
    label_dict = {n: M.network.nodes[n][labels] for n in M.network.nodes().keys()}
    nx.draw_networkx_labels(M.network, pos=pos, labels=label_dict)

    if flux.any():
        rounded = np.round(flux, decimals=2)
        edgeflux = sparse.dok_matrix(rounded)
        # Draw edge labels
        nx.draw_networkx_edge_labels(M.network, pos=pos, edge_labels=edgeflux,
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
