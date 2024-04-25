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
    ax.plot((x1, x2), (y1, y2), color="black", lw=5)
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
        **kwargs,
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
        **kwargs,
    )


def dense2scatter_plot(
    ax,
    dense,
    point_size=5,
    alpha=1.0,
    marker=".",
    time_step=1e-3,
    jitter=None,
    double=False,
    color_list=["black", "black"],
    **kwargs,
):
    n = dense.shape[1] // 2
    if double:
        ras0 = datasets.dense2ras(dense[:, :n], time_step)
        ras1 = datasets.dense2ras(dense[:, n:], time_step)
        ras = [ras0, ras1]
    else:
        ras = [datasets.dense2ras(dense, time_step)]
    for r, c in zip(ras, color_list):
        if len(r):
            noise = np.zeros(r[:, 0].shape)
            if jitter is not None:
                noise = jitter * np.random.randn(*r[:, 0].shape)
            ax.scatter(
                r[:, 0] + noise,
                r[:, 1],
                s=point_size,
                alpha=alpha,
                marker=marker,
                color=c,
                **kwargs,
            )


def save_plots(fileprefix, extensions=["pdf", "png"], dpi=300):
    """Apply savefig function to multiple extensions"""
    for ext in extensions:
        plt.savefig("%s.%s" % (fileprefix, ext), dpi=dpi, bbox_inches="tight")


def turn_axis_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_activity_over_trials(
    model,
    data,
    ax,
    layer_idx=0,
    neuron_idx=0,
    nb_trials=51,
    marker=".",
    point_size=5,
    point_alpha=1,
    color="black",
    nolabel=False,
):
    # Run model once and get activities
    scores = model.evaluate(data, one_batch=True).tolist()

    hidden_groups = model.groups[1:-1]
    hid_activity = [
        g.get_flattened_out_sequence().detach().cpu().numpy() for g in hidden_groups
    ]

    nb_trials = np.min((nb_trials, hid_activity[layer_idx].shape[0]))

    act = hid_activity[layer_idx][:, :, neuron_idx]
    for i in range(nb_trials):
        spikes = np.where(act[i])[0]
        ax.scatter(
            spikes,
            np.ones_like(spikes) * i,
            color=color,
            alpha=point_alpha,
            marker=marker,
            s=point_size,
        )

    if not nolabel:
        ax.set_ylabel("Trial idx")
        ax.set_xlabel("Time (s)")
    ax.set_xlim(-3, model.nb_time_steps + 3)
    ax.set_ylim(-3, nb_trials + 3)
    ax.set_xticks([0, model.nb_time_steps])
    ax.set_yticks([0, nb_trials - 1])
    sns.despine()


def plot_activity(
    model,
    data,
    nb_samples=2,
    figsize=(5, 5),
    dpi=250,
    marker=".",
    point_size=5,
    point_alpha=1,
    pal=sns.color_palette("muted", n_colors=20),
    bg_col="#AAAAAA",
    bg_col2="#DDDDDD",
    pos=(0, 0),
    off=(0, -0.05),
    labels=[7, 8],
):
    # Run model once and get activities
    scores = model.evaluate(data, one_batch=True).tolist()

    inp = model.input_group.get_flattened_out_sequence().detach().cpu().numpy()
    hidden_groups = model.groups[1:-1]
    hid_activity = [
        g.get_flattened_out_sequence().detach().cpu().numpy() for g in hidden_groups
    ]
    out_group = model.out.detach().cpu().numpy()

    n = model.nb_inputs

    inps = [inp]

    nb_groups = len(hidden_groups)
    nb_total_units = np.sum([g.nb_units for g in hidden_groups])
    hr = [1] + [4 * g.nb_units / nb_total_units for g in hidden_groups] + [1]
    hr = list(reversed(hr))  # since we are plotting from bottom to top

    fig, ax = plt.subplots(
        nb_groups + 2,
        nb_samples,
        figsize=figsize,
        dpi=dpi,
        sharex="row",
        sharey="row",
        gridspec_kw={"height_ratios": hr},
    )

    sns.despine()

    samples = []

    label_idx = 0
    while len(samples) < nb_samples:
        for i in range(len(data)):
            if data[i][1] == labels[label_idx]:
                label_idx += 1
                samples.append(i)
                break

    for i, s in enumerate(samples):
        # plot and color input spikes

        for idx, inp in enumerate(inps):
            c = pal[i]

            ax[-1][i].scatter(
                np.where(inp[s])[0],
                np.where(inp[s])[1] + idx * n,
                s=point_size,
                marker=marker,
                color=c,
                alpha=point_alpha,
            )
            # invert y-axis
            ax[-1][i].invert_yaxis()
        ax[-1][i].set_ylim(-3, model.nb_inputs + 3)
        ax[-1][i].set_xlim(-3, model.nb_time_steps + 3)

        if i != 0:
            ax[-1][i].set_yticks([])
            ax[-1][i].spines["left"].set_visible(False)
            ax[-1][i].spines["right"].set_visible(False)
            ax[-1][i].spines["top"].set_visible(False)

        # plot hidden layer spikes
        for g in range(nb_groups):
            ax[-(2 + g)][i].scatter(
                np.where(hid_activity[g][s])[0],
                np.where(hid_activity[g][s])[1],
                s=point_size / 2,
                marker=marker,
                color="k",
                alpha=point_alpha,
            )

            ax[-(2 + g)][0].set_ylabel("Hid. " + str(g))
            # turn off x-axis
            ax[-(2 + g)][i].set_xticks([])
            ax[-(2 + g)][i].set_yticks([])
            ax[-(2 + g)][i].spines["bottom"].set_visible(False)
            ax[-(2 + g)][i].set_xlim(-3, model.nb_time_steps + 3)

            if i != 0:
                turn_axis_off(ax[-(2 + g)][i])

        for line_index, ro_line in enumerate(np.transpose(out_group[s])):
            c = bg_col
            alpha = 0.5
            zorder = -5

            for j, sidx in enumerate(samples):
                if line_index == data[sidx][1]:
                    c = pal[j]
                    alpha = 1
                    zorder = 1
            ax[0][i].plot(ro_line, color=c, zorder=zorder, alpha=alpha)

        ax[-1][i].set_xlabel("Time (s)")
        if i != 0:
            turn_axis_off(ax[0][i])

        # invert y-axis
        ax[-1][i].invert_yaxis()

    ax[0][0].set_xticks([])
    ax[0][0].spines["bottom"].set_visible(False)

    ax[-1][0].set_ylabel("Input")
    ax[0][0].set_ylabel("Readout")
    ax[-1][0].set_yticks([])
    ax[0][0].set_yticks([])

    duration = round(model.nb_time_steps * model.time_step * 10) / 10
    ax[-1][0].set_xticks([0, model.nb_time_steps], [0, duration])

    plt.tight_layout()


def plot_activity_snapshot(
    model,
    data,
    nb_samples=5,
    figsize=(10, 5),
    dpi=250,
    marker=".",
    point_size=5,
    point_alpha=1,
    pal=sns.color_palette("muted", n_colors=20),
    bg_col="#AAAAAA",
    bg_col2="#DDDDDD",
    double=False,
    pos=(0, -1),
    off=(0, -0.05),
    title=False,
):
    print("plotting snapshot")

    # Run model once and get activities
    scores = model.evaluate(data, one_batch=True).tolist()

    inp = model.input_group.get_flattened_out_sequence().detach().cpu().numpy()
    hidden_groups = model.groups[1:-1]
    hid_activity = [
        g.get_flattened_out_sequence().detach().cpu().numpy() for g in hidden_groups
    ]
    out_group = model.out.detach().cpu().numpy()

    n = model.nb_inputs
    m = out_group.shape[-1]
    if double:
        n = n // 2
        m = m // 2

    if double:
        inp1 = inp[:, :, :n]
        inp2 = inp[:, :, n:]
        inps = [inp1, inp2]
    else:
        inps = [inp]

    nb_groups = len(hidden_groups)
    nb_total_units = np.sum([g.nb_units for g in hidden_groups])
    hr = [1] + [4 * g.nb_units / nb_total_units for g in hidden_groups] + [1]
    hr = list(reversed(hr))  # since we are plotting from bottom to top

    fig, ax = plt.subplots(
        nb_groups + 2,
        nb_samples,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": hr},
    )

    for i in range(nb_samples):
        # plot and color input spikes
        for idx, inp in enumerate(inps):
            if double:
                c = pal[data[idx][i][1] + idx * m]
            else:
                c = pal[data[i][1]]
            ax[-1][i].scatter(
                np.where(inp[i])[0],
                np.where(inp[i])[1] + idx * n,
                s=point_size,
                marker=marker,
                color=c,
                alpha=point_alpha,
            )
        ax[-1][i].set_ylim(-3, model.nb_inputs + 3)
        ax[-1][i].set_xlim(-3, model.nb_time_steps + 3)
        turn_axis_off(ax[-1][i])

        # plot hidden layer spikes
        for g in range(nb_groups):
            ax[-(2 + g)][i].scatter(
                np.where(hid_activity[g][i])[0],
                np.where(hid_activity[g][i])[1],
                s=point_size / 2,
                marker=marker,
                color="k",
                alpha=point_alpha,
            )
            turn_axis_off(ax[-(2 + g)][i])

            ax[-(2 + g)][0].set_ylabel("Hid. " + str(g))

        for line_index, ro_line in enumerate(np.transpose(out_group[i])):
            if double:
                if line_index == data[0][i][1] or line_index == data[1][i][1] + m:
                    ax[0][i].plot(ro_line, color=pal[line_index])
                else:
                    if line_index < m:
                        ax[0][i].plot(ro_line, color=bg_col, zorder=-5, alpha=0.5)
                    else:
                        ax[0][i].plot(ro_line, color=bg_col2, zorder=-5, alpha=0.5)
            else:
                if line_index == data[i][1]:
                    ax[0][i].plot(ro_line, color=pal[line_index])
                else:
                    ax[0][i].plot(ro_line, color=bg_col, zorder=-5, alpha=0.5)
            if title:
                ax[0][i].set_title(data[i][1])
            turn_axis_off(ax[0][i])

        # invert y-axis
        ax[-1][i].invert_yaxis()

    dur_50 = 50e-3 / model.time_step
    # print(dur_10)
    add_xscalebar(ax[-1][0], dur_50, label="50ms", pos=pos, off=off, fontsize=8)

    ax[-1][0].set_ylabel("Input")
    ax[0][0].set_ylabel("Readout")
    plt.tight_layout()


def plot_activity_snapshot_old(
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
                color_list=[color],
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
                # color="black",
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
