import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def convert_size(width, height):
    """
    Convert figure size from mm to inches.

    Parameters
    ----------
    width : float
        The width in mm.
    height : float
        The height in mm.

    Returns
    -------
    tuple
        The width and height in inches.
    """

    width = width / 25.4
    height = height / 25.4

    return (width, height)


def colored_heatmap(data,
                    figsize,
                    labels,
                    fontsizes,
                    ticks,
                    bounds,
                    colormap='magma',
                    filename=None,
                    dpi=300,
                    filetype='svg',
                    fraction=0.0454):
    """
    Create a heatmap.

    Parameters
    ----------
    arnold_tongue : array_like
        The Arnold tongue.
    figsize : tuple
        The figure size in inches.
    labels : tuple
        The labels for the x and y axes, and the colorbar.
    fontsizes : tuple
        The label and tick font sizes.
    ticks : tuple
        The ticks for the x and y axes.
    bounds : tuple
        The lower and upper bounds for the colormap.
    colormap : str, optional
        The colormap for the plot. The default is 'magma'.
    filename : str, optional
        The filename to save the plot. The default is None.
    dpi : int, optional
        The resolution of the plot. The default is 300.
    filetype : str, optional
        The file type for the plot. The default is 'svg'.
    fraction : float, optional
        The fraction of the original axes to use for the colorbar. The default
        is 0.0454.

    Returns
    -------
    None
    """

    figure = plt.figure(figsize=figsize,
                        dpi=dpi if filetype != 'svg' else None)
    sns.set_style('white')
    sns.set_context('paper')
    sns.set_palette('muted')

    xlabel, ylabel, cbar_label = labels
    label_fontsize, tick_fontsize, cbar_labelsize = fontsizes
    xticks, yticks = ticks

    im = plt.imshow(data, cmap=colormap, vmin=bounds[0], vmax=bounds[1])

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    if cbar_label is not None:
        cbar = plt.colorbar(im, fraction=fraction)
        cbar.ax.tick_params(labelsize=tick_fontsize)

        cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=cbar_labelsize)

    num_ticks = len(xticks)
    xtick_locations = np.linspace(0, data.shape[1] - 1, num_ticks)
    num_ticks = len(yticks)
    ytick_locations = np.linspace(0, data.shape[0] - 1, num_ticks)

    plt.xticks(xtick_locations, xticks, fontsize=tick_fontsize)
    plt.yticks(ytick_locations, yticks, fontsize=tick_fontsize)

    plt.tight_layout()

    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)
    else:
        plt.show()
