import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.mach import Mach


def test_mach_basic():
    z = np.linspace(0, 10, 5)
    res = Mach(10.0, 10.0, z, 0.3, 0.0, 0.2, 340.0, 0.0, 293.15, 287.0, 1.4, 0.0)
    assert len(res) == 3
    for arr in res:
        assert arr.shape == (5, 1)
        assert np.all(np.isfinite(arr))

    mach_vals = res[0].ravel()
    deriv = np.gradient(mach_vals, z)
    assert np.all(np.isfinite(deriv))
