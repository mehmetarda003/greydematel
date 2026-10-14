"""Gri sayı (aralık) işlemleri."""

from __future__ import annotations

import numpy as np


def grey_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """İki gri matrisi veya vektörü toplar: [l,u] + [l,u]."""
    return a + b


def grey_scalar_div(interval: np.ndarray, s: float) -> np.ndarray:
    """Gri aralığı skaler S'ye böler."""
    if s == 0:
        raise ValueError("Normalleştirme paydası S sıfır olamaz.")
    return interval / s


def matrix_row_sums_grey(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Satır toplamları (R): alt ve üst limit."""
    return lower.sum(axis=1), upper.sum(axis=1)


def matrix_col_sums_grey(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sütun toplamları (C): alt ve üst limit."""
    return lower.sum(axis=0), upper.sum(axis=0)


def grey_interval_add(
    r_l: np.ndarray, r_u: np.ndarray, c_l: np.ndarray, c_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """R_i + C_i gri toplamı (her kriter için)."""
    return r_l + c_l, r_u + c_u


def grey_interval_subtract(
    r_l: np.ndarray, r_u: np.ndarray, c_l: np.ndarray, c_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """R − C gri çıkarma: alt = R_alt − C_üst, üst = R_üst − C_alt."""
    return r_l - c_u, r_u - c_l


def defuzzify_mid(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Ağırlık / skor için orta nokta (l+u)/2."""
    return (lower + upper) / 2.0


def normalize_weights(mid_values: np.ndarray) -> np.ndarray:
    """Pozitif değerleri toplam 1 olacak şekilde normalize eder."""
    v = np.maximum(mid_values, 0.0)
    total = v.sum()
    if total <= 0:
        return np.ones_like(v) / len(v)
    return v / total
