"""Grey DEMATEL — Adım 1–5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grey_dematel.grey_ops import (
    defuzzify_mid,
    grey_interval_subtract,
    grey_scalar_div,
    matrix_col_sums_grey,
    matrix_row_sums_grey,
    normalize_weights,
)


@dataclass
class GreyDematelResult:
    """Hesaplama çıktıları."""

    A_lower: np.ndarray
    A_upper: np.ndarray
    S: float
    D_lower: np.ndarray
    D_upper: np.ndarray
    T_lower: np.ndarray
    T_upper: np.ndarray
    T_crisp: np.ndarray
    R: np.ndarray
    C: np.ndarray
    prominence: np.ndarray
    relation: np.ndarray
    weights: np.ndarray
    criterion_names: list[str]
    # Referans: T üzerinden gri R,C ve doğru aralık R−C (özet tabloda kullanılmaz)
    R_grey_lower: np.ndarray
    R_grey_upper: np.ndarray
    C_grey_lower: np.ndarray
    C_grey_upper: np.ndarray
    relation_grey_lower: np.ndarray
    relation_grey_upper: np.ndarray


def _total_relation_matrix(d_lower: np.ndarray, d_upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = d_lower.shape[0]
    i_mat = np.eye(n)
    try:
        inv_l = np.linalg.inv(i_mat - d_lower)
        inv_u = np.linalg.inv(i_mat - d_upper)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Toplam ilişki matrisi hesaplanamadı (I-D) tekil. "
            "Değerlendirme matrisini kontrol edin."
        ) from e
    t_lower = d_lower @ inv_l
    t_upper = d_upper @ inv_u
    return t_lower, t_upper


def _crisp_rc_from_total(T_lower: np.ndarray, T_upper: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """T matrisini orta nokta ile durulaştır; R, C, R+C, R−C ve ağırlık girdisini üret."""
    t_crisp = defuzzify_mid(T_lower, T_upper)
    r = t_crisp.sum(axis=1)
    c = t_crisp.sum(axis=0)
    prominence = r + c
    relation = r - c
    return t_crisp, r, c, prominence, relation


def compute_grey_dematel(
    A_lower: np.ndarray,
    A_upper: np.ndarray,
    criterion_names: list[str] | None = None,
) -> GreyDematelResult:
    """
    Gri DEMATEL adımlarını uygular.

    A ve D gri kalır; T elde edildikten sonra (alt+üst)/2 durulaştırması ile
    R, C, R+C, R−C ve ağırlıklar kesin (crisp) değerlerden hesaplanır.
    """
    A_lower = np.asarray(A_lower, dtype=float)
    A_upper = np.asarray(A_upper, dtype=float)
    if A_lower.shape != A_upper.shape or A_lower.ndim != 2:
        raise ValueError("A_lower ve A_upper aynı boyutta kare matris olmalı.")
    n = A_lower.shape[0]
    if criterion_names is None:
        criterion_names = [f"K{i + 1}" for i in range(n)]
    if len(criterion_names) != n:
        raise ValueError("Kriter adı sayısı matris boyutu ile eşleşmeli.")

    if np.any(A_lower > A_upper):
        raise ValueError("Alt limit üst limitten büyük olamaz.")
    if not np.allclose(np.diag(A_lower), 0) or not np.allclose(np.diag(A_upper), 0):
        raise ValueError("Köşegen elemanları [0,0] olmalı.")

    row_sum_u = A_upper.sum(axis=1).max()
    col_sum_u = A_upper.sum(axis=0).max()
    S = float(max(row_sum_u, col_sum_u))
    if S <= 0:
        raise ValueError("Normalleştirme için S > 0 gerekir; matris tamamen sıfır olamaz.")

    D_lower = grey_scalar_div(A_lower, S)
    D_upper = grey_scalar_div(A_upper, S)

    T_lower, T_upper = _total_relation_matrix(D_lower, D_upper)

    T_crisp, R, C, prominence, relation = _crisp_rc_from_total(T_lower, T_upper)
    weights = normalize_weights(prominence)

    R_grey_lower, R_grey_upper = matrix_row_sums_grey(T_lower, T_upper)
    C_grey_lower, C_grey_upper = matrix_col_sums_grey(T_lower, T_upper)
    relation_grey_lower, relation_grey_upper = grey_interval_subtract(
        R_grey_lower, R_grey_upper, C_grey_lower, C_grey_upper
    )

    return GreyDematelResult(
        A_lower=A_lower,
        A_upper=A_upper,
        S=S,
        D_lower=D_lower,
        D_upper=D_upper,
        T_lower=T_lower,
        T_upper=T_upper,
        T_crisp=T_crisp,
        R=R,
        C=C,
        prominence=prominence,
        relation=relation,
        weights=weights,
        criterion_names=criterion_names,
        R_grey_lower=R_grey_lower,
        R_grey_upper=R_grey_upper,
        C_grey_lower=C_grey_lower,
        C_grey_upper=C_grey_upper,
        relation_grey_lower=relation_grey_lower,
        relation_grey_upper=relation_grey_upper,
    )


def default_threshold(T_lower: np.ndarray, T_upper: np.ndarray) -> float:
    """Eşik: T matrisi orta noktalarının ortalaması (köşegen hariç)."""
    mid = (T_lower + T_upper) / 2.0
    n = mid.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(mid[mask].mean())


def influence_edges(
    T_crisp: np.ndarray,
    threshold: float,
    names: list[str],
) -> list[tuple[str, str, float]]:
    """Eşiği aşan yönlü kenarlar (i -> j)."""
    n = T_crisp.shape[0]
    edges: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = T_crisp[i, j]
            if w >= threshold:
                edges.append((names[i], names[j], w))
    return edges
