# -*- coding: utf-8 -*-
"""
03_calidad.py — Calidad de datos y diccionario reconstruido (sección 4.2 + entregable 1).

  - % de nulos por columna (clasificación usable / condicional / descartable)
  - matriz/heatmap de ausencia de datos
  - inconsistencias: cardinalidad y variantes categóricas, fechas imposibles,
    montos negativos/cero, monedas
  - diccionario de datos reconstruido (tipo, rol, disponibilidad, %null, cardinalidad)
  Genera reports/04_calidad_datos.md, reports/05_diccionario_datos.md y figuras.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import config as C
import utils as U

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATE_COLS = ["cargo_ready", "atd", "ata", "ie_date", "fecha_num_dam", "fecha_descarga",
             "fecha_tarja", "fecha_levante", "senasa_inspeccion", "senasa_liberacion",
             "demurrage_exp", "fecha_retiro_t1", "fecha_entrega_base_t2", "fecha_fin_viaje",
             "receipt_confirmation", "fecha_liquidacion", "fecha_registro_gastos",
             "quote_request_date", "assignment_date", "pick_up_date"]
NUM_COLS = ["qty", "amount", "amount_usd", "ctnr_qty", "bulks", "peso_bruto",
            "payment_term", "target_usd", "target_usd_sin_trib", "igv_usd",
            "n_oc_lineas", "n_oc_distintas", "n_productos", "n_conceptos"]


def col_kind(df, c):
    if c in DATE_COLS or pd.api.types.is_datetime64_any_dtype(df[c]):
        return "fecha"
    if c in NUM_COLS or pd.api.types.is_numeric_dtype(df[c]):
        return "numérico"
    return "categórico"


def main():
    m = pd.read_parquet(os.path.join(C.PROC_DIR, "operaciones_modelables.parquet"))
    op = pd.read_parquet(os.path.join(C.PROC_DIR, "operativo_lineas.parquet"))

    # ----------------------------------------------------------------- #
    # 4.2.9 — % nulos y clasificación
    # ----------------------------------------------------------------- #
    R = U.MdReport("4.2 — Calidad de datos (dataset modelable: 1 fila/operación)")
    cols = [c for c in m.columns if c not in ("op_id_full", "op_key")]
    nul = pd.DataFrame({
        "columna": cols,
        "pct_nulos": [100 * m[c].isna().mean() for c in cols],
        "n_unicos": [m[c].nunique(dropna=True) for c in cols],
        "tipo": [col_kind(m, c) for c in cols],
    }).sort_values("pct_nulos", ascending=False)

    def clasif(p):
        return "descartable" if p > 80 else ("condicional" if p > 30 else "usable")
    nul["clasificación"] = nul["pct_nulos"].map(clasif)

    R.h("% de nulos por columna (ordenado)")
    R.p(nul.assign(pct_nulos=nul["pct_nulos"].round(1)).to_markdown(index=False))
    R.h("Resumen por clasificación")
    R.p(nul["clasificación"].value_counts().rename_axis("clasificación")
        .reset_index(name="n_columnas").to_markdown(index=False))
    R.p("\n- **usable** (<30% nulos): listas para modelar.\n"
        "- **condicional** (30–80%): aplican sólo a un subconjunto (p. ej. datos de "
        "contenedor sólo en marítimo, peso bruto sólo 2025-2026, fechas SENASA sólo plantas).\n"
        "- **descartable** (>80%): demasiado vacías para usarse crudas.")

    # ----------------------------------------------------------------- #
    # 4.2.10 — heatmap de ausencia
    # ----------------------------------------------------------------- #
    show = nul.sort_values("pct_nulos")["columna"].tolist()
    mat = m[show].isna().astype(int)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(mat.T.values, aspect="auto", cmap="rocket_r", interpolation="nearest")
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show, fontsize=6)
    ax.set_xlabel("operaciones (ordenadas por archivo/fecha)")
    ax.set_title("4.2.10 — Matriz de ausencia (oscuro = nulo)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f03_nulos_heatmap.png")); plt.close(fig)

    # nulos en contenedor vs modo (¿coinciden con aéreo?)
    if "ctnr_qty" in m and "mode" in m:
        modo = m["mode"].map(U.norm_text)
        cont_null_air = m.loc[modo.str.contains("air", na=False), "ctnr_qty"].isna().mean()
        cont_null_sea = m.loc[modo.str.contains("sea", na=False), "ctnr_qty"].isna().mean()
        R.h("Patrón de ausencia: contenedor vs modo de transporte")
        R.p(f"- `ctnr_qty` nulo en **AIR**: {100*cont_null_air:.0f}%  |  en **SEA**: {100*cont_null_sea:.0f}%")
        R.p("Confirma que los datos de contenedor faltan estructuralmente en aéreo (no es error).")

    # ----------------------------------------------------------------- #
    # 4.2.11 — inconsistencias
    # ----------------------------------------------------------------- #
    R.h("Inconsistencias detectadas")

    # categóricas: cardinalidad y variantes por normalización
    cat_check = ["mode", "type", "incoterm", "categoria", "status", "moneda",
                 "country_origin", "pol", "pod", "um"]
    rows = []
    for c in cat_check:
        if c not in m:
            continue
        raw_u = m[c].dropna().nunique()
        norm_u = m[c].dropna().map(U.norm_text).nunique()
        ejemplos = ", ".join(map(str, m[c].dropna().unique()[:8]))
        rows.append(dict(columna=c, n_crudos=raw_u, n_normalizados=norm_u,
                         variantes_por_formato=raw_u - norm_u, ejemplos=ejemplos[:90]))
    R.p("**Cardinalidad y variantes categóricas** (crudos vs normalizados: la diferencia "
        "son duplicados por mayúsculas/acentos/espacios):")
    R.p(pd.DataFrame(rows).to_markdown(index=False))

    # fechas imposibles
    R.h("Coherencia temporal")
    bad = []
    if "atd" in m and "ata" in m:
        d = (m["ata"] - m["atd"]).dt.days
        n_neg = (d < 0).sum(); n_ok = d.notna().sum()
        bad.append(("arribo (ata) antes de zarpe (atd)", int(n_neg), int(n_ok)))
        R.p(f"- Tránsito ata−atd: válidos={n_ok}, **imposibles (<0)={n_neg}**, "
            f"mediana={d[d>=0].median():.0f} días, p95={d[d>=0].quantile(.95):.0f} días")
    if "ata" in m and "fecha_levante" in m:
        d2 = (m["fecha_levante"] - m["ata"]).dt.days
        R.p(f"- Días hasta levante (levante−ata): mediana={d2[d2>=0].median():.0f}, "
            f"imposibles(<0)={int((d2<0).sum())}")

    # montos
    R.h("Montos")
    for c in ["amount_usd", "target_usd"]:
        if c in m:
            R.p(f"- `{c}`: ≤0 = {int((m[c]<=0).sum())}, nulos = {int(m[c].isna().sum())}, "
                f"negativos = {int((m[c]<0).sum())}")
    if "moneda" in m:
        R.p(f"- Monedas de mercadería presentes: {m['moneda'].dropna().map(U.norm_text).unique().tolist()}")

    R.save(os.path.join(C.REPORTS_EDA, "04_calidad_datos.md"))

    # ----------------------------------------------------------------- #
    # Entregable 1 — Diccionario de datos reconstruido
    # ----------------------------------------------------------------- #
    D = U.MdReport("Entregable 1 — Diccionario de datos reconstruido")
    D.p("Reconstruido desde los CSV crudos. `rol`: id / feature / leakage. "
        "`dispo` = disponible al momento de la llegada (para evitar leakage).")
    drows = []
    for c in cols:
        meta = C.VAR_META.get(c, ("—", "—", "—", ""))
        rol, dispo, imp, desc = meta
        kind = col_kind(m, c)
        if kind == "numérico" and m[c].notna().any():
            rango = f"[{np.nanmin(m[c]):.0f} … {np.nanmax(m[c]):.0f}]"
        elif kind == "fecha" and m[c].notna().any():
            rango = f"[{m[c].min():%Y-%m} … {m[c].max():%Y-%m}]"
        else:
            top = m[c].dropna().astype(str).value_counts().head(3).index.tolist()
            rango = ", ".join(top)[:60]
        drows.append(dict(variable=c, tipo=kind, rol=rol, dispo=dispo,
                          importancia=imp, pct_nulos=round(100*m[c].isna().mean(), 1),
                          n_unicos=m[c].nunique(dropna=True),
                          rango_o_ejemplos=rango, descripcion=desc))
    dd = pd.DataFrame(drows)
    # ordenar: features primero por importancia, luego leakage, luego id
    rank = {"alta": 0, "media": 1, "baja": 2, "—": 3}
    rolrank = {"feature": 0, "id": 1, "leakage": 2, "—": 3}
    dd["_r"] = dd["rol"].map(rolrank).fillna(3) * 10 + dd["importancia"].map(rank).fillna(3)
    dd = dd.sort_values("_r").drop(columns="_r")
    D.h("Variables canónicas")
    D.p(dd.to_markdown(index=False))
    D.h("Variables marcadas como LEAKAGE (excluir como features)")
    leak = dd[dd["rol"] == "leakage"]["variable"].tolist()
    D.p(", ".join(f"`{x}`" for x in leak) if leak else "—")
    D.save(os.path.join(C.REPORTS_EDA, "05_diccionario_datos.md"))

    print("nulos -> 04_calidad_datos.md ; diccionario -> 05_diccionario_datos.md")
    print("clasif:", nul["clasificación"].value_counts().to_dict())


if __name__ == "__main__":
    main()
