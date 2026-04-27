import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .arrays import to_np


def visualize_minimal_pde(
    u_controlled,
    f_pred,
    save_path,
    sample_id,
):

    u_controlled = to_np(u_controlled)
    f_pred = to_np(f_pred)

    vmin_state = u_controlled.min()
    vmax_state = u_controlled.max()
    vmin_control = f_pred.min()
    vmax_control = f_pred.max()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 0.03]},
    )

    ax_top, cax_top = axes[0, 0], axes[0, 1]
    ax_bot, cax_bot = axes[1, 0], axes[1, 1]

    im1 = ax_top.imshow(
        u_controlled, aspect="auto", cmap="RdBu_r", vmin=vmin_state, vmax=vmax_state
    )
    ax_top.set_title("Controlled State u", fontsize=18)
    ax_top.set_xlabel("Spatial position", fontsize=16)
    ax_top.set_ylabel("Temporal step", fontsize=16)
    ax_top.tick_params(axis="both", which="major", labelsize=14)
    cbar1 = fig.colorbar(im1, cax=cax_top)
    cbar1.ax.tick_params(labelsize=11)

    im2 = ax_bot.imshow(
        f_pred, aspect="auto", cmap="viridis", vmin=vmin_control, vmax=vmax_control
    )
    ax_bot.set_title("Predicted Control f", fontsize=18)
    ax_bot.set_xlabel("Spatial position", fontsize=16)
    ax_bot.set_ylabel("Temporal step", fontsize=16)
    ax_bot.tick_params(axis="both", which="major", labelsize=14)
    cbar2 = fig.colorbar(im2, cax=cax_bot)
    cbar2.ax.tick_params(labelsize=11)

    try:
        fig.set_constrained_layout_pads(
            w_pad=0.02, h_pad=0.04, hspace=0.06, wspace=0.02
        )
    except Exception:
        pass

    ax_top.yaxis.set_label_coords(-0.04, 0.5)
    ax_bot.yaxis.set_label_coords(-0.04, 0.5)
    ax_top.xaxis.set_label_coords(0.5, -0.12)
    ax_bot.xaxis.set_label_coords(0.5, -0.12)

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"visualization_{sample_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_state_slices(
    u_controlled,
    safety_upper_profile,
    safety_lower_profile,
    save_path,
    sample_id,
    times=(0.3, 0.5, 0.8),
):

    u = to_np(u_controlled)
    ub = to_np(safety_upper_profile)
    lb = to_np(safety_lower_profile)

    idxs = []
    for t in times:
        idx = int(np.floor(t * 10.0 + 0.5))
        idx = max(0, min(10, idx))
        idxs.append(idx)

    N = u.shape[-1]
    x = (np.arange(1, N + 1, dtype=np.float32)) / (N + 1)

    y_min = np.inf
    y_max = -np.inf
    for k in idxs:
        y_min = min(y_min, u[k].min(), lb[k].min())
        y_max = max(y_max, u[k].max(), ub[k].max())
    y_range = float(y_max - y_min) if np.isfinite(y_max - y_min) else 1.0
    y_pad = 0.1 * y_range if y_range > 0 else 0.05
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.2), constrained_layout=True, sharey=True, sharex=True
    )

    x_min, x_max = float(x.min()), float(x.max())

    for ax, k, t_label in zip(axes, idxs, times):
        ax.plot(x, u[k], color="black", lw=2.0, label="State u")
        ax.plot(x, ub[k], color="tab:red", lw=1.5, ls="--", label="Upper bound")
        ax.plot(x, lb[k], color="tab:blue", lw=1.5, ls="--", label="Lower bound")
        ax.set_title(f"t={t_label}", fontsize=32)
        ax.set_xlabel("x", fontsize=32)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="major", labelsize=32)
        ax.grid(alpha=0.2)
        ax.set_xlim(x_min, x_max)
        ax.margins(x=0)
        ax.set_ylim(y_lo, y_hi)

    axes[0].set_ylabel("u", fontsize=32)

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="upper right", fontsize=20)

    try:
        fig.set_constrained_layout_pads(
            w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.04
        )
    except Exception:
        pass

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"slices_{sample_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
