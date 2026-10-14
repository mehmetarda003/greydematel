"""Grey DEMATEL — Streamlit arayüzü."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from grey_dematel.core import compute_grey_dematel, default_threshold, influence_edges
from grey_dematel.scales import GREY_SCALE, SCALE_LABELS

st.set_page_config(page_title="Grey DEMATEL", page_icon="📊", layout="wide")

st.title("Grey DEMATEL Analizi")
st.caption(
    "Gri sayılı direkt ilişki matrisi → normalleştirme → toplam ilişki → "
    "R/C toplamları → eşik ve etki-yönlü graf"
)


def matrices_from_scale_grid(n: int, names: list[str], key_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros((n, n))
    upper = np.zeros((n, n))
    st.markdown("**Direkt ilişki matrisi (A)** — hücrede dil skalasını seçin. Köşegen otomatik [0,0].")
    cols = st.columns(min(n, 6))
    for j in range(n):
        with cols[j % len(cols)]:
            st.markdown(f"**→ {names[j]}**")
    for i in range(n):
        row_cols = st.columns(n)
        for j in range(n):
            with row_cols[j]:
                if i == j:
                    st.selectbox(
                        f"{names[i]}→{names[j]}",
                        ["[0,0]"],
                        index=0,
                        disabled=True,
                        key=f"{key_prefix}_diag_{i}_{j}",
                        label_visibility="collapsed",
                    )
                else:
                    choice = st.selectbox(
                        f"{names[i]}→{names[j]}",
                        SCALE_LABELS,
                        index=0,
                        key=f"{key_prefix}_cell_{i}_{j}",
                        label_visibility="collapsed",
                    )
                    lo, hi = GREY_SCALE[choice]
                    lower[i, j] = lo
                    upper[i, j] = hi
    return lower, upper


def build_results_excel(result) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        n = len(result.criterion_names)
        idx = result.criterion_names

        def grey_df(lo, hi):
            return pd.DataFrame(
                [[f"[{lo[i,j]:.4f}, {hi[i,j]:.4f}]" for j in range(n)] for i in range(n)],
                index=idx,
                columns=idx,
            )

        grey_df(result.A_lower, result.A_upper).to_excel(writer, sheet_name="A_Direkt")
        pd.DataFrame({"S": [result.S]}).to_excel(writer, sheet_name="S_Normalize", index=False)
        grey_df(result.D_lower, result.D_upper).to_excel(writer, sheet_name="D_Normalize")
        grey_df(result.T_lower, result.T_upper).to_excel(writer, sheet_name="T_Toplam")

        pd.DataFrame(
            result.T_crisp,
            index=idx,
            columns=idx,
        ).to_excel(writer, sheet_name="T_Durulastirilmis")

        summary = pd.DataFrame(
            {
                "Kriter": idx,
                "R (crisp)": result.R,
                "C (crisp)": result.C,
                "R+C": result.prominence,
                "R-C": result.relation,
                "Ağırlık (normalize)": result.weights,
                "Grup": [
                    "Neden (Cause)" if result.relation[k] >= 0 else "Sonuç (Effect)"
                    for k in range(n)
                ],
                "R-C (gri alt)": result.relation_grey_lower,
                "R-C (gri üst)": result.relation_grey_upper,
            }
        )
        summary.to_excel(writer, sheet_name="R_C_Ozet", index=False)
    return buf.getvalue()


def plot_cause_effect(result, ax):
    ax.scatter(result.relation, result.prominence, s=120, c="#2563eb", zorder=3)
    for k, name in enumerate(result.criterion_names):
        ax.annotate(
            name,
            (result.relation[k], result.prominence[k]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_xlabel("R − C (ilişki: neden ↔ sonuç)")
    ax.set_ylabel("R + C (önem)")
    ax.set_title("Neden–Sonuç Diyagramı")
    ax.grid(True, alpha=0.3)


def plot_influence_graph(edges, names, ax):
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    if g.number_of_edges() == 0:
        ax.text(0.5, 0.5, "Eşik üzerinde kenar yok", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    pos = nx.spring_layout(g, seed=42, k=1.2)
    widths = [1.0 + 2.0 * g[u][v]["weight"] for u, v in g.edges()]
    nx.draw_networkx_nodes(g, pos, node_color="#e0e7ff", edgecolors="#4338ca", node_size=900, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(g, pos, width=widths, edge_color="#64748b", arrows=True, arrowsize=16, ax=ax)
    ax.set_title("Etki-yönlü graf (eşik üstü ilişkiler)")
    ax.set_axis_off()


with st.sidebar:
    st.header("Girdi")
    mode = st.radio("Veri girişi", ["Arayüzden matris", "Dosyadan yükle"], index=0)
    n = st.number_input("Kriter sayısı (n)", min_value=2, max_value=12, value=4, step=1)
    default_names = [f"K{i + 1}" for i in range(int(n))]
    names_text = st.text_area("Kriter adları (satır başına bir)", value="\n".join(default_names), height=120)
    names = [line.strip() for line in names_text.splitlines() if line.strip()]
    if len(names) != int(n):
        st.warning(f"{int(n)} kriter için tam {int(n)} ad girin. Şimdilik K1…Kn kullanılıyor.")
        names = default_names

st.subheader("Dil skalası")
st.table(
    pd.DataFrame(
        [{"Ifade": k, "Alt limit": v[0], "Üst limit": v[1]} for k, v in GREY_SCALE.items()]
    )
)

A_lower: np.ndarray | None = None
A_upper: np.ndarray | None = None

if mode == "Arayüzden matris":
    A_lower, A_upper = matrices_from_scale_grid(int(n), names, "ui")
else:
    st.markdown("**Alt limit** ve **üst limit** için `n×n` CSV veya Excel yükleyin (ilk satır/sütun kriter adları).")
    col1, col2 = st.columns(2)
    with col1:
        file_l = st.file_uploader("Alt limit matrisi", type=["csv", "xlsx"], key="fl")
    with col2:
        file_u = st.file_uploader("Üst limit matrisi", type=["csv", "xlsx"], key="fu")
    if file_l and file_u:
        from grey_dematel.io_csv import read_matrix_pair

        try:
            A_lower, A_upper, file_names = read_matrix_pair(file_l.getvalue(), file_u.getvalue())
            if len(file_names) == A_lower.shape[0]:
                names = file_names
            st.success(f"Matris boyutu: {A_lower.shape[0]}×{A_lower.shape[1]}")
        except Exception as e:
            st.error(str(e))

run = st.button("Hesapla", type="primary")

if run and A_lower is not None and A_upper is not None:
    try:
        result = compute_grey_dematel(A_lower, A_upper, names)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.success(f"Normalleştirme paydası **S** = {result.S:.6f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Özet", "Matrisler", "Grafikler", "İndir"])

    with tab1:
        st.caption(
            "R, C, R+C, R−C ve ağırlıklar: T matrisi (alt+üst)/2 durulaştırması sonrası kesin değerler."
        )
        df = pd.DataFrame(
            {
                "Kriter": result.criterion_names,
                "R": result.R,
                "C": result.C,
                "R+C": result.prominence,
                "R−C": result.relation,
                "Ağırlık": result.weights,
                "Grup": [
                    "Neden" if result.relation[k] >= 0 else "Sonuç"
                    for k in range(len(names))
                ],
            }
        )
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown("**D (normalleştirilmiş)** orta nokta")
        d_mid = (result.D_lower + result.D_upper) / 2
        st.dataframe(pd.DataFrame(d_mid, index=names, columns=names))
        st.markdown("**T (toplam ilişki)** — durulaştırılmış `(alt+üst)/2`")
        st.dataframe(pd.DataFrame(result.T_crisp, index=names, columns=names))

    with tab3:
        thr_default = default_threshold(result.T_lower, result.T_upper)
        threshold = st.slider(
            "Eşik değeri (etki-yönlü graf)",
            min_value=0.0,
            max_value=float(max(thr_default * 2, 0.01)),
            value=float(thr_default),
            step=float(thr_default / 50) if thr_default > 0 else 0.001,
        )
        edges = influence_edges(result.T_crisp, threshold, names)
        st.caption(f"Eşik ≥ {threshold:.4f} → {len(edges)} yönlü ilişki")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_cause_effect(result, axes[0])
        plot_influence_graph(edges, names, axes[1])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        xlsx = build_results_excel(result)
        st.download_button(
            "Sonuçları Excel indir",
            data=xlsx,
            file_name="grey_dematel_sonuc.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

elif run:
    st.error("Önce geçerli bir matris girin veya dosya yükleyin.")
