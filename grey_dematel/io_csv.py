"""CSV / Excel okuma."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def read_matrix_pair(
    lower_bytes: bytes | None,
    upper_bytes: bytes | None,
    sheet_lower: str | int = 0,
    sheet_upper: str | int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    İki dosyada (veya tek Excel'de iki sayfa) n×n sayısal matris okur.
    İlk satır/sütun kriter adları olabilir; yoksa K1, K2, ... kullanılır.
    """
    if lower_bytes is None or upper_bytes is None:
        raise ValueError("Alt ve üst limit dosyaları gerekli.")

    def _load(data: bytes, sheet) -> pd.DataFrame:
        bio = io.BytesIO(data)
        if data[:2] == b"PK":
            return pd.read_excel(bio, sheet_name=sheet, header=0, index_col=0)
        return pd.read_csv(bio, header=0, index_col=0)

    df_l = _load(lower_bytes, sheet_lower)
    df_u = _load(upper_bytes, sheet_upper)
    if df_l.shape != df_u.shape:
        raise ValueError("Alt ve üst matris boyutları aynı olmalı.")

    names = [str(c) for c in df_l.columns.tolist()]
    lower = df_l.to_numpy(dtype=float)
    upper = df_u.to_numpy(dtype=float)
    return lower, upper, names
