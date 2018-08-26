from .utils import *
import matplotlib.pyplot as plt


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

def plot_clusters(network, clusters, scale=1):
    spm = shortest_path_matrix(network)
    shells = get_shells(network, clusters, spm)
    shell_pos = get_shell_pos(network, shells, scale=scale)

    return shell_pos


