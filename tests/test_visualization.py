import os
import sys
import numpy as np
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.visualization import plot_starter_field, plot_xy_slice, plot_directivity_mask, plot_directivity_3d

def test_plot_functions_return_figures():
    U0 = np.ones((5,5))
    y = np.linspace(0,1,5)
    z = np.linspace(0,1,5)
    fig = plot_starter_field(U0, y, z, 0.5, 0.5, 100)
    assert isinstance(fig, Figure)

    D = np.ones((5,5))
    fig2 = plot_directivity_mask(D, y, z, 0.5, 0.5, 1.0, 100)
    assert isinstance(fig2, Figure)

    L = np.ones((5,5,5))
    x = np.linspace(0,1,5)
    fig_xy, fig_line = plot_xy_slice(L, x, y, 0, 0, 100, 'SPL', True, 1.0, 0.0, 0.0, 0.0)
    assert isinstance(fig_xy, Figure)
    assert isinstance(fig_line, Figure)

    Phi = np.zeros((5,5))
    Theta = np.zeros((5,5))
    mag = np.ones((5,5))
    fig3d = plot_directivity_3d(Phi, Theta, mag)
    assert isinstance(fig3d, Figure)