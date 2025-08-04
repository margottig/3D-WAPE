import numpy as np
from dataclasses import dataclass

@dataclass
class RiwDefinition:
    riw_field_ex: np.ndarray  # shape (n_freq, n_phi, n_theta)
    index_500hz: int = 0


def berechne_lphi_ex(phi, theta, data, phi_res=5, theta_res=5):
    """Return directivity level for given angles using bilinear interpolation.

    Parameters
    ----------
    phi : float
        Azimuth angle in degrees.
    theta : float
        Elevation angle in degrees.
    data : ndarray
        Array of shape (n_freq, n_phi, n_theta) containing directivity values in
        dB. Angles are assumed to be sampled at regular ``phi_res`` and
        ``theta_res`` increments starting at 0 degrees.
    phi_res : int, optional
        Resolution of the ``phi`` grid in degrees. Default is 5.
    theta_res : int, optional
        Resolution of the ``theta`` grid in degrees. Default is 5.

    Returns
    -------
    ndarray
        Interpolated values for each frequency.
    """
    phi = np.clip(phi, 0, 360)
    theta = np.clip(theta, 0, 180)

    # Compute surrounding grid indices
    phi_idx = phi / phi_res
    theta_idx = theta / theta_res
    i_phi = int(np.floor(phi_idx)) % (data.shape[1] - 1)
    i_theta = int(np.floor(theta_idx))
    xsi = phi_idx - np.floor(phi_idx)
    eta = theta_idx - np.floor(theta_idx)

    w11 = data[:, i_phi, i_theta]
    w21 = data[:, i_phi + 1, i_theta]
    w22 = data[:, i_phi + 1, i_theta + 1]
    w12 = data[:, i_phi, i_theta + 1]

    LPhiTerz = (
        (1 - xsi) * (1 - eta) * w11
        + xsi * (1 - eta) * w21
        + xsi * eta * w22
        + (1 - xsi) * eta * w12
    )

    return LPhiTerz