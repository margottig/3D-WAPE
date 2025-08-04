import matplotlib.pyplot as plt
import numpy as np
from modules.visualization import *


def plot_starter_field(U0_3D_initial_plane, y_coords, z_coords, y_center, z_height, freq):
    """Plot starter field amplitude."""
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(
        np.abs(U0_3D_initial_plane),
        aspect="auto",
        origin="lower",
        extent=[y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],
        interpolation="bilinear",
    )
    plt.colorbar(label="Amplitude |U0|")
    plt.xlabel(f"Y (m), source at {y_center:.1f}m")
    plt.ylabel("Z (m)")
    # plt.title(f"Initial Field Amp (Salomons+XHN {freq}Hz)")
    if y_coords[0] <= y_center <= y_coords[-1] and z_coords[0] <= z_height <= z_coords[-1]:
        plt.scatter(y_center, z_height, c="red", marker="x", s=100, label="Source Center")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_xy_slice(L_pp3D, x_vals, y_vals, iy_slice_idx, iz_slice_idx, freq, title_str_spl, CALIBRATE_OUTPUT, dim_x, plot_y_val, plot_z_val, z_r_receiver_height):
    """Plot XY slice pcolormesh and line plot from 3D data."""
    # n_isolines = 10
    fig_xy, ax_xy = plt.subplots(figsize=(7, 6))
    if (
        L_pp3D.ndim == 3
        and L_pp3D.size > 0
        and 0 <= iz_slice_idx < L_pp3D.shape[0]
        and x_vals.size > 1
        and y_vals.size > 1
    ):
        data_to_plot_xy = L_pp3D[iz_slice_idx, :, :]        
        
        vmax_plot_xy = (
            np.nanmax(data_to_plot_xy)
            if not np.all(np.isnan(data_to_plot_xy))
            else 100
        )
        if np.isfinite(vmax_plot_xy):
            if vmax_plot_xy < 30:
                vmin_plot_xy = vmax_plot_xy - 70
            else:
                vmin_plot_xy = max(30, vmax_plot_xy - 70)  #70
        else:
            vmin_plot_xy = 30
            
        if vmax_plot_xy <= vmin_plot_xy:
            vmax_plot_xy = vmin_plot_xy + 1
        im_xy = ax_xy.pcolormesh(
            x_vals,
            y_vals,
            data_to_plot_xy,
            shading="gouraud",
            cmap="magma",
            vmin=vmin_plot_xy,
            vmax=vmax_plot_xy,
        )

        cb_xy = fig_xy.colorbar(im_xy, ax=ax_xy)
        cb_xy.set_label("SPL (dB re 20µPa)" if CALIBRATE_OUTPUT else "Relative SPL (dB)")

        # overlay contour lines
        levels = np.arange(30, 91, 5)   # e.g. every 5 dB from 40 to 70
        cs = ax_xy.contour(
            x_vals,
            y_vals,
            data_to_plot_xy,
            levels=levels,
            colors='white',
            linewidths=1.0
        )
        ax_xy.clabel(cs, fmt='%d dB', inline=True, fontsize=8)
    
    
    else:
        ax_xy.text(
            0.5,
            0.5,
            "XY SPL slice unavailable",
            transform=ax_xy.transAxes,
            ha="center",
            va="center",
        )
    ax_xy.set_xlabel("Range x (m)")
    ax_xy.set_ylabel("Lateral y (m)")
    ax_xy.set_title(f"{title_str_spl} in X-Y plane at z ≈ {plot_z_val:.2f}m ({freq:.0f} Hz)")
    fig_line, ax_line = plt.subplots(figsize=(7, 6))
    line_data_available = False
    if (
        L_pp3D.ndim == 3
        and L_pp3D.size > 0
        and 0 <= iz_slice_idx < L_pp3D.shape[0]
        and 0 <= iy_slice_idx < L_pp3D.shape[1]
        and L_pp3D.shape[2] == len(x_vals)
    ):
        line_data = L_pp3D[iz_slice_idx, iy_slice_idx, :]
        ax_line.plot(x_vals, line_data, linewidth=1.5, label=f"{freq:.0f} Hz SPL")
        
        min_plot_val = (
            np.nanmin(line_data) if not np.all(np.isnan(line_data)) else 30.0
        )
        max_plot_val = (
            np.nanmax(line_data) if not np.all(np.isnan(line_data)) else 100.0
        )
        min_ylim = (
            (min_plot_val if np.isfinite(min_plot_val) else 30.0) - 10
        )
        max_ylim = (
            (max_plot_val if np.isfinite(max_plot_val) else 100.0) + 10
        )
        if max_ylim <= min_ylim:
            max_ylim = min_ylim + 1
        ax_line.set_ylim([min_ylim, max_ylim])   
        
        ax_line.legend()
        line_data_available = True
    if not line_data_available:
        ax_line.text(0.5, 0.5, "Line plot SPL data unavailable", ha="center", va="center", transform=ax_line.transAxes)
        ax_line.set_ylim([30, 120])
    ax_line.set_xlim([0, dim_x if dim_x > 0 else 1])
    ax_line.grid(True)
    ax_line.set_xlabel("Range x (m)")
    ax_line.set_ylabel("SPL (dB re 20µPa)" if CALIBRATE_OUTPUT else "Relative SPL (dB)")
    ax_line.set_title(
        f"{title_str_spl} at y ≈ {plot_y_val:.1f}m, z ≈ {plot_z_val:.2f}m (target z_rec={z_r_receiver_height:.1f}m), ({freq:.0f} Hz)"
    )
    plt.tight_layout()
    plt.show(block=False)
    return fig_xy, fig_line

__all__ = [
    "plot_starter_field",
    "plot_xy_slice",
    "plot_directivity_3d",
]