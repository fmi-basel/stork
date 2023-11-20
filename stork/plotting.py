import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import datasets


def add_scalebar(
    ax, extent=(1.0, 0.0), pos=(0.0, 0.0), off=(0.0, 0.0), label=None, **kwargs
):
    scale = np.concatenate((np.diff(ax.get_xlim()), np.diff(ax.get_ylim())))
    x1, y1 = np.array(pos) * scale
    x2, y2 = np.array((x1, y1)) + np.array(extent)
    xt, yt = np.array((np.mean((x1, x2)), np.mean((y1, y2)))) + np.array(off) * scale
    ax.plot((x1, x2), (y1, y2), color="black")
    if label:
        ax.text(xt, yt, label, **kwargs)


def add_xscalebar(ax, length, label=None, pos=(0.0, -0.1), off=(0.0, -0.07), **kwargs):
    add_scalebar(
        ax,
        label=label,
        extent=(length, 0.0),
        pos=pos,
        off=off,
        verticalalignment="top",
        horizontalalignment="center",
        **kwargs
    )


def add_yscalebar(ax, length, label=None, pos=(-0.1, 0.0), off=(-0.07, 0.0), **kwargs):
    add_scalebar(
        ax,
        label=label,
        extent=(0.0, length),
        pos=pos,
        off=off,
        verticalalignment="center",
        horizontalalignment="left",
        rotation=90,
        **kwargs
    )


def dense2scatter_plot(
    ax,
    dense,
    point_size=5,
    alpha=1.0,
    marker=".",
    time_step=1e-3,
    jitter=None,
    **kwargs
):
    ras = datasets.dense2ras(dense, time_step)
    if len(ras):
        noise = np.zeros(ras[:, 0].shape)
        if jitter is not None:
            noise = jitter * np.random.randn(*ras[:, 0].shape)
        ax.scatter(
            ras[:, 0] + noise,
            ras[:, 1],
            s=point_size,
            alpha=alpha,
            marker=marker,
            **kwargs
        )


def save_plots(fileprefix, extensions=["pdf", "png"], dpi=300):
    """Apply savefig function to multiple extensions"""
    for ext in extensions:
        plt.savefig("%s.%s" % (fileprefix, ext), dpi=dpi, bbox_inches="tight")


def plot_activity_snapshot(
    model,
    data=None,
    labels=None,
    pred=None,
    nb_samples=5,
    plot_groups=None,
    marker=".",
    point_size=5,
    point_alpha=1.0,
    time_jitter=None,
    random_samples=False,
    show_predictions=False,
    readout_threshold=None,
    show_input_class=True,
    input_heatmap=False,
    pal=None,
    n_colors=20,
):
    """Plot an activity snapshot

    Args:
        model (nn.Module): The model
        data (data, optional): Data to send thorugh the network. Defaults to None.
        labels (labels, optional): Lables of the given data. Defaults to None.
        pred (vector of int, optional): Instead of data, we can also pass predictions directly. Defaults to None.
        nb_samples (int, optional): Number of samples to plot. Defaults to 5.
        plot_groups (model.groups, optional): The hidden groups to show. If None, all hidden groups are shown. Defaults to None.
        marker (str, optional): Marker type for raster plots. Defaults to ".".
        point_size (int, optional): Raster plot point size. Defaults to 5.
        point_alpha (float, optional): Raster plot alpha. Defaults to 1.0.
        time_jitter (float, optional): Adds Gaussian noise of amplitude `time_jitter` to spike times to remove Moiree effect in plots due to discrete time. Defaults to None.
        random_samples (bool, optional): Select random samples from model output tensor. Defaults to False.
        show_predictions (bool, optional): Prints text boxes with output predictions in output field. Defaults to False.
        readout_threshold (float, optional): Plots a threshold (line) in the readout neurons. Defaults to None.
        show_input_class (bool, optional): Color the input spikes in the color corresponding to label. Defaults to True.
        input_heatmap (bool, optional): Plot input as heatmap instead of raster plot. Defaults to False.
        pal (color Palette, optional): Color Palette. Defaults to None.
        n_colors (int, optional): Number of different classes (colors). Defaults to 20.
    """

    if data is not None and labels is None:
        labels = [d[1] for d in data]
    if data is not None:
        pred = model.predict(data)

    nb_batches = len(data) // model.batch_size
    if len(data) // model.batch_size < len(data) / model.batch_size:
        size_of_last_batch = len(data) - nb_batches * model.batch_size
        nb_batches += 1
    else:
        size_of_last_batch = model.batch_size

    pred = pred[-size_of_last_batch:]
    if labels is not None:
        n_colors = len(np.unique(labels))
        labels = labels[-size_of_last_batch:]

    time_step = model.time_step

    if plot_groups is None:
        hidden_groups = model.groups[1:-1]
    else:
        hidden_groups = plot_groups

    nb_groups = len(hidden_groups)
    nb_total_units = np.sum([g.nb_units for g in hidden_groups])
    hr = [1] + [4 * g.nb_units / nb_total_units for g in hidden_groups] + [1]
    hr = list(reversed(hr))  # since we are plotting from bottom to top
    gs = GridSpec(2 + nb_groups, nb_samples, height_ratios=hr)

    in_group = model.input_group.get_flattened_out_sequence().detach().cpu().numpy()
    hid_groups = [
        g.get_flattened_out_sequence().detach().cpu().numpy() for g in hidden_groups
    ]
    out_group = model.out.detach().cpu().numpy()
    idx = np.arange(len(in_group))

    if random_samples:
        np.random.shuffle(idx)

    text_props = {"ha": "center", "va": "center", "fontsize": 8}
    for i in range(nb_samples):
        if i == 0:
            a0 = ax = plt.subplot(gs[i + (nb_groups + 1) * nb_samples])
        else:
            ax = plt.subplot(gs[i + (nb_groups + 1) * nb_samples], sharex=a0, sharey=a0)

        k = idx[i]
        color = "black"

        # COLOR CHOICES
        if pal is None:
            if n_colors <= 10:
                colors = [
                    "#CC6677",
                    "#332288",
                    "#DDCC77",
                    "#117733",
                    "#88CCEE",
                    "#882255",
                    "#44AA99",
                    "#999933",
                    "#AA4499",
                    "#EE8866",
                ]
                pal = sns.color_palette(colors, n_colors=n_colors)

            else:
                pal = sns.color_palette("muted", n_colors=n_colors)

        # Colored input class
        if show_input_class and data is not None:
            clipped = np.clip(labels, 0, len(pal) - 1)
            color = pal[int(clipped[k])]
        else:
            color = "black"

        if not input_heatmap:
            dense2scatter_plot(
                ax,
                in_group[k],
                marker=marker,
                point_size=point_size,
                alpha=point_alpha,
                time_step=time_step,
                color=color,
                jitter=time_jitter,
            )
        else:
            shape = in_group[k].shape
            ax.imshow(
                in_group[k].T,
                aspect="auto",
                origin="lower",
                extent=(0, shape[0] * time_step, 0, shape[1]),
            )
        ax.axis("off")

        # Plot scatter plots
        if i == 0:
            ax.text(
                -0.15,
                0.5,
                "Input",
                text_props,
                color="black",
                transform=ax.transAxes,
                fontsize=8,
                rotation=90,
            )
            add_xscalebar(ax, 10e-3, label="10ms", pos=(0.0, -0.2), fontsize=8)

        for h in range(nb_groups):
            ax = plt.subplot(gs[i + (nb_groups - h) * nb_samples], sharex=a0)
            dense2scatter_plot(
                ax,
                hid_groups[h][k],
                marker=marker,
                point_size=point_size,
                alpha=point_alpha,
                time_step=time_step,
                color="black",
                jitter=time_jitter,
            )
            ax.axis("off")
            if i == 0:
                label = "Hidden"
                if hidden_groups[h].name is not None:
                    label = hidden_groups[h].name
                else:
                    if nb_groups > 1:
                        label = "Hid. %i" % (h + 1)
                ax.text(
                    -0.15,
                    0.5,
                    label,
                    text_props,
                    color="black",
                    transform=ax.transAxes,
                    fontsize=8,
                    rotation=90,
                )

        # Readout neurons
        if i == 0:
            ax0out = ax = plt.subplot(gs[i], sharex=a0)
            ax.text(
                -0.15,
                0.5,
                "Readout",
                text_props,
                color="black",
                transform=ax.transAxes,
                fontsize=8,
                rotation=90,
            )
        else:
            ax = plt.subplot(gs[i], sharex=a0, sharey=ax0out)

        times = np.arange(len(out_group[k])) * time_step

        for line_index, ro_line in enumerate(np.transpose(out_group[k])):
            if labels is not None:
                if line_index != int(labels[k]):
                    color = "#DDDDDD"
                    zorder = 5
                else:
                    color = pal[line_index]
                    zorder = 10
            else:
                color = "black"
            ax.plot(times, ro_line, color=color, zorder=zorder, lw=1)

        if readout_threshold is not None:
            ax.axhline(readout_threshold, alpha=1.0, color="black", ls="dashed", lw=0.5)

        if show_predictions:
            ax.text(
                0.5,
                0.8,
                "Pred: %i" % pred[k],
                color="black",
                transform=ax.transAxes,
                fontsize=8,
            )

        ax.set_xlabel("Time (s)")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Readout ampl.")

    plt.tight_layout()
    sns.despine()
