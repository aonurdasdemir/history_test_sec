"""This file includes functions for plotting the results and optimization related data."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from copy import deepcopy
import numpy as np
from mpl_toolkits import axes_grid1
import matplotlib
import pandas as pd

# matplotlib.use("TkAgg")


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def show_field(
    field,
    cmap=None,
    vmin=None,
    vmax=None,
    label="",
    show_colorbar=True,
    ax=None,
    use_log_scale=False,
    outline=False,
    outline_val=None,
    title="",
    file_save_name=None,
):
    """Shows the physical field using standard plotting options."""
    if ax is None:
        _, ax = plt.subplots(1, tight_layout=True, figsize=(5, 5))

    normalizer_factory = LogNorm if use_log_scale else Normalize

    # Plot the transpose of the field to account for the fact that dim1=X, dim2=Y.
    h = ax.imshow(field.T, cmap=cmap, origin="lower", norm=normalizer_factory(vmin, vmax))

    if show_colorbar:
        add_colorbar(h)

    if outline:
        ax.contour(outline_val, levels=0, linewidths=1, colors="k")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    if file_save_name is not None:
        plt.savefig(file_save_name + "eps.png")
    plt.show()


def show_device(eps, outline=False, show_colorbar=True, ax=None, vmin=None, vmax=None, title="", file_save_name=None):
    """Shows the physical device."""
    eps_r = np.hstack([deepcopy(eps)])

    return show_field(
        np.abs(eps_r),
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        label="",
        show_colorbar=show_colorbar,
        ax=ax,
        use_log_scale=False,
        outline=outline,
        outline_val=np.abs(eps_r).transpose(),
        title=title,
        file_save_name=file_save_name,
    )


def visualize_penalty(length_penalty, preprocessed_density, fname):
    """Shows the image of penalty functions."""
    penalty_value = length_penalty((preprocessed_density.reshape((-1, 1)), True, 0))
    gs_array = length_penalty._gs_array
    gv_array = length_penalty._gv_array
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    show_device(preprocessed_density, ax=ax1, show_colorbar=False)
    show_field(gs_array, ax=ax2)
    show_field(gv_array, ax=ax3)
    plt.tight_layout()
    plt.savefig(fname + "penalties.png")


def objective_plot(file_name, ax2, ax3):
    """Plots the the objective function and penalty functions w.r.t. iterations.

    Indicates the scale jumps with dashed orange line and projection strength changes with dotted blue lines.
    Args:
        file_name: File name for reading the optimization data.
        ax2: Subplots axis to draw the objective function.
        ax3: Subplots axis to draw the penalty functions.
    """
    csv_data = pd.read_csv(file_name + "opt_data.csv")
    iters = np.array(csv_data["iteration"])
    obj = np.array(csv_data["obj"])
    penalty_cv = np.array(csv_data["penalty_cv"])
    penalty_cs = np.array(csv_data["penalty_cs"])

    cell_size = np.array(csv_data["cell_size (um)"])
    beta = np.array(csv_data["beta1"])
    current_cell_size = cell_size[0]
    size_change_iters = []
    i = 0
    for size in cell_size:
        i += 1
        if size != current_cell_size:
            size_change_iters.append(i)
            current_cell_size = size
    seen = set()
    res = []
    beta = beta.tolist()
    for i, n in enumerate(beta):
        if n not in seen:
            res.append(i)
            seen.add(n)
    ax2.semilogy(iters, obj, label="loss")
    if len(size_change_iters) > 0:
        ax2.plot(
            np.ones(50) * size_change_iters[0], np.linspace(0, np.amax(obj), 50), linestyle="--", label="scale change"
        )
    ax2.vlines(res, ymin=0, ymax=np.amax(obj), linestyle="dotted")
    ax2.legend()
    ax2.set_xlabel("Iterations")
    ax3.semilogy(iters, penalty_cv, label="g_v")
    ax3.semilogy(iters, penalty_cs, label="g_s")
    ax3.legend()

