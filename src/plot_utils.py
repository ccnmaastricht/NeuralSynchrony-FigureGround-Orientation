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

    sns.set_style('white')
    sns.set_context('paper')
    sns.set_palette('muted')

    plt.figure(figsize=figsize, dpi=dpi if filetype != 'svg' else None)

    title, xlabel, ylabel, cbar_label = labels
    title_fontsize, label_fontsize, tick_fontsize, cbar_labelsize = fontsizes
    xticks, yticks = ticks

    im = plt.imshow(data, cmap=colormap, vmin=bounds[0], vmax=bounds[1])

    plt.title(title, fontsize=title_fontsize)
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

        plt.close()
    else:
        plt.show()


def plot_dAIC(session,
              dAIC,
              figsize,
              labels,
              fontsizes,
              line_color,
              text_color,
              fill,
              filename=None,
              dpi=300,
              filetype='svg'):
    """
    Plot dAIC against session with specific formatting.

    Parameters
    ----------
    session : array_like
        The session numbers.
    dAIC : array_like
        The delta AIC values.
    figsize : tuple
        The figure size in inches.
    labels : tuple
        The labels for the title, x and y axes, and the fill labels.
    fontsizes : tuple
        The title and label font sizes.
    line_color : str
        The color for the line plot.
    text_color : str
        The color for the text.
    fill : tuple
        The fill colors, alpha, x and y bounds, fill labels, and fill label
        positions.
    filename : str, optional
        The filename to save the plot. The default is None.
    dpi : int, optional
        The resolution of the plot. The default is 300.
    filetype : str, optional
        The file type for the plot. The default is 'svg'.

    Returns
    -------
    None
    """
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette('muted')

    plt.figure(figsize=figsize, dpi=dpi if filetype != 'svg' else None)

    title, xlabel, ylabel = labels
    title_fontsize, label_fontsize, tick_fontsize = fontsizes
    fill_colors, fill_alpha, x_bounds, y_bounds, fill_labels, fill_label_x, fill_label_y = fill

    plt.plot(session, dAIC, marker='o', linestyle='-', color=line_color)
    for bounds, fill_color, label, position_x, position_y in zip(
            x_bounds, fill_colors, fill_labels, fill_label_x, fill_label_y):
        plt.fill_between(bounds,
                         y_bounds[0],
                         y_bounds[1],
                         color=fill_color,
                         alpha=fill_alpha)
        plt.text(position_x,
                 position_y,
                 label,
                 fontsize=label_fontsize,
                 color=text_color)

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)

        plt.close()
    else:
        plt.show()
