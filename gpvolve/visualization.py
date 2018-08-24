import matplotlib.pyplot as plt


def plot_timescales(timescales, figsize=None, n=None, color='orange'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i for i in range(0, len(timescales[:n]))], timescales[:n], color=color)
    ax.set_title("Timescales")
    return fig, ax

def plot_eigenvalues(timescales):
    pass

def plot_clusters(network, clusters):
    pass

