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
    None or Figure
        If `filename` is None, return the figure.
    """

    sns.set_style('white')
    sns.set_context('paper')
    sns.set_palette('muted')

    figure = plt.figure(figsize=figsize,
                        dpi=dpi if filetype != 'svg' else None)

    title, xlabel, ylabel, cbar_label = labels
    title_fontsize, label_fontsize, tick_fontsize, cbar_labelsize = fontsizes
    xticks, yticks = ticks

    x_range = np.max(xticks) - np.min(xticks)
    y_range = np.max(yticks) - np.min(yticks)
    aspect_ratio = x_range / y_range

    im = plt.imshow(data,
                    cmap=colormap,
                    vmin=bounds[0],
                    vmax=bounds[1],
                    extent=[xticks[0], xticks[-1], yticks[-1], yticks[0]],
                    aspect=aspect_ratio)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    if cbar_label is not None:
        cbar = plt.colorbar(im, fraction=fraction)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=cbar_labelsize)

    plt.xticks(xticks, fontsize=tick_fontsize)
    plt.yticks(yticks, fontsize=tick_fontsize)

    plt.tight_layout()

    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)
        plt.close()
    else:
        return figure


def comparative_lineplot(x,
                         y,
                         bounds,
                         figsize,
                         labels,
                         fontsizes,
                         line_color,
                         filename=None,
                         dpi=300,
                         filetype='svg'):
    """
    Plot a line plot with shaded error bounds.

    Parameters
    ----------
    x : array_like
        The x values.
    y : array_like  
        The y values.
    bounds : tuple
        The lower and upper bounds for the shaded region.
    figsize : tuple
        The figure size in inches.
    labels : tuple
        The labels for the title, x and y axes.
    fontsizes : tuple
        The title, label, and tick font sizes.
    line_color : str
        The color for the line plot.
    filename : str, optional
        The filename to save the plot. The default is None.
    dpi : int, optional
        The resolution of the plot. The default is 300.
    filetype : str, optional
        The file type for the plot. The default is 'svg'.

    Returns
    -------
    None or Figure
        If `filename` is None, return the figure.
    """

    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette('muted')

    figure = plt.figure(figsize=figsize)

    label, xlabel, ylabel = labels
    label_fontsize, tick_fontsize = fontsizes

    plt.plot(x,
             y[0],
             marker='o',
             linestyle='-',
             color=line_color[0],
             label=label[0])
    plt.fill_between(x,
                     bounds[0][0],
                     bounds[1][0],
                     color=line_color[0],
                     alpha=0.2)

    plt.plot(x,
             y[1],
             marker='o',
             linestyle='-',
             color=line_color[1],
             label=label[1])
    plt.fill_between(x,
                     bounds[0][1],
                     bounds[1][1],
                     color=line_color[1],
                     alpha=0.2)

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    plt.legend(fontsize=label_fontsize)

    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)
        plt.close()
    else:
        return figure


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
    None or Figure
        If `filename` is None, return the figure.
    """
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette('muted')

    figure = plt.figure(figsize=figsize,
                        dpi=dpi if filetype != 'svg' else None)

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
        return figure


def fit_barplot(mean_fit,
                sem_fit,
                sessions,
                noise_ceiling,
                figsize,
                labels,
                fontsizes,
                capsize,
                face_color,
                alpha=0.3,
                filename=None,
                dpi=300,
                filetype='svg'):

    figure = plt.figure(figsize=figsize,
                        dpi=dpi if filetype != 'svg' else None)

    title, xlabel, ylabel = labels
    title_fontsize, label_fontsize, tick_fontsize = fontsizes

    # Create a bar plot
    plt.bar(sessions,
            mean_fit,
            yerr=1.96 * sem_fit,
            capsize=capsize,
            color=face_color)

    # set the min and max of the y-axis
    plt.ylim(0, 1)

    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Add gray regions above each bar to indicate the noise ceiling
    for i, session in enumerate(sessions):
        x_values = np.array([session - 0.4, session + 0.4])
        y1_values = np.full_like(x_values, noise_ceiling[i][0])
        y2_values = np.full_like(x_values, noise_ceiling[i][1])

        plt.fill_between(x_values,
                         y1_values,
                         y2=y2_values,
                         color='gray',
                         alpha=alpha)

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()

    # Save the figure
    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)
        plt.close()
    else:
        return figure
