"""Temel Grey DEMATEL testleri."""

import numpy as np
import pytest

from grey_dematel.core import compute_grey_dematel
from grey_dematel.grey_ops import grey_interval_subtract


def test_small_matrix_runs():
    n = 3
    lo = np.zeros((n, n))
    hi = np.zeros((n, n))
    lo[0, 1], hi[0, 1] = 1, 2
    lo[1, 2], hi[1, 2] = 2, 3
    lo[2, 0], hi[2, 0] = 0, 1
    r = compute_grey_dematel(lo, hi)
    assert r.S > 0
    assert r.weights.sum() == pytest.approx(1.0)
    assert r.T_lower.shape == (n, n)
    assert r.T_crisp.shape == (n, n)
    assert np.allclose(r.R, r.T_crisp.sum(axis=1))
    assert np.allclose(r.C, r.T_crisp.sum(axis=0))
    assert np.allclose(r.prominence, r.R + r.C)
    assert np.allclose(r.relation, r.R - r.C)


def test_grey_interval_subtraction_order():
    r_l = np.array([0.5])
    r_u = np.array([1.0])
    c_l = np.array([0.2])
    c_u = np.array([0.8])
    lo, hi = grey_interval_subtract(r_l, r_u, c_l, c_u)
    assert lo[0] == pytest.approx(r_l[0] - c_u[0])
    assert hi[0] == pytest.approx(r_u[0] - c_l[0])
    assert lo[0] <= hi[0]


def test_grey_rc_interval_valid_in_result():
    lo = np.zeros((2, 2))
    hi = np.zeros((2, 2))
    lo[0, 1], hi[0, 1] = 1, 3
    lo[1, 0], hi[1, 0] = 0, 2
    r = compute_grey_dematel(lo, hi)
    assert np.all(r.relation_grey_lower <= r.relation_grey_upper)
