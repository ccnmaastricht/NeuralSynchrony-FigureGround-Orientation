import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_arnold_tongue(arnold_tongue, figsize, labels, fontsizes, ticks, show = True, save = False, filename = None, dpi=300, fraction=0.0454):
    '''
    Plot the Arnold tongue.

    Parameters
    ----------
    arnold_tongue : array_like
        The Arnold tongue.
    figsize : tuple
        The figure size in inches.
    labels : tuple
        The labels for the title, x and y axes, and the colorbar.
    fontsizes : tuple
        The title, label, and tick font sizes.
    ticks : tuple
        The ticks for the x and y axes.
    show : bool, optional
        Whether to show the plot. The default is True.
    save : bool, optional
        Whether to save the plot. The default is False.
    filename : str, optional
        The filename to save the plot. The default is None.
    dpi : int, optional
        The resolution of the plot. The default is 300.

    Returns
    -------
    None
    '''

    figure = plt.figure(figsize=figsize, dpi=dpi)
    sns.set_style('white')
    sns.set_context('paper')
    sns.set_palette('muted')

    
    title, xlabel, ylabel, cbar_label = labels
    title_fontsize, label_fontsize, tick_fontsize, cbar_labelsize = fontsizes
    xticks, yticks = ticks

    im = plt.imshow(arnold_tongue, cmap='jet')

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    cbar = plt.colorbar(im, fraction=fraction)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=cbar_labelsize)

    num_ticks = len(xticks)
    xtick_locations = np.linspace(0, arnold_tongue.shape[1] - 1, num_ticks)
    num_ticks = len(yticks)
    ytick_locations = np.linspace(0, arnold_tongue.shape[0] - 1, num_ticks)

    plt.xticks(xtick_locations, xticks, fontsize=tick_fontsize)
    plt.yticks(ytick_locations, yticks, fontsize=tick_fontsize)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=dpi)

    if show:
        plt.show()