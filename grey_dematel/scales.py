"""Gri DEMATEL dil skalası."""

from __future__ import annotations

GREY_SCALE: dict[str, tuple[float, float]] = {
    "Etkisi Yok [0,0]": (0.0, 0.0),
    "Etkisi Az [0,1]": (0.0, 1.0),
    "Etkisi Orta [1,2]": (1.0, 2.0),
    "Etkisi Çok [2,3]": (2.0, 3.0),
    "Etkisi Çok Fazla [3,4]": (3.0, 4.0),
}

SCALE_LABELS = list(GREY_SCALE.keys())
