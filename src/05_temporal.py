# -*- coding: utf-8 -*-
"""
05_temporal.py — Variables derivadas y análisis temporal (sección 4.5).

  - Tiempos del proceso (tránsito, numeración, levante, depósito, llegada a fundo)
    y su relación con el costo (más días en depósito ⇒ más sobrestadía/almacenaje).
  - Estacionalidad y tendencia temporal (inflación de tarifas) ⇒ valida usar
    validación temporal (forward chaining), no K-fold aleatorio.
  - Catálogo de variables derivadas propuestas.
Genera reports/08_temporal.md y figuras f06_*.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils as U
import dataset as D

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
TGT = "target_servicios"


def main():
    df = D.load(fiable_only=True)
    # traer componente de sobrestadía/almacenaje por operación
    piv = pd.read_parquet(os.path.join(C.PROC_DIR, "target_por_concepto.parquet"))
    keep = [c for c in ["op_id_full", "SOBRESTADIA_ALMACENAJE", "FLETE_INTERNACIONAL",
                        "TRANSPORTE_T2_LIMA_FUNDO"] if c in piv.columns]
    df = df.merge(piv[keep], on="op_id_full", how="left")

    R = U.MdReport(f"4.5 — Tiempos del proceso, estacionalidad y derivadas (n={len(df)})")

    # ---------- tiempos derivados ----------
    R.h("Tiempos del proceso (días)")
    tcols = ["transit_days", "dias_num_dam", "dias_levante", "dias_deposito", "dias_a_fundo"]
    desc = df[tcols].describe(percentiles=[.5, .9]).T[["count", "mean", "50%", "90%", "max"]]
    R.p(desc.round(1).to_markdown())

    R.h("Relación días en depósito ↔ sobrestadía/almacenaje")
    if "SOBRESTADIA_ALMACENAJE" in df:
        d = df[["dias_deposito", "SOBRESTADIA_ALMACENAJE", "demurrage_exp"]].copy()
        d = d.dropna(subset=["dias_deposito"])
        sp = stats.spearmanr(d["dias_deposito"], d["SOBRESTADIA_ALMACENAJE"].fillna(0))[0]
        con = (d["SOBRESTADIA_ALMACENAJE"].fillna(0) > 0)
        R.p(f"- Spearman(días depósito, costo sobrestadía) = **{sp:.2f}** (n={len(d)})")
        R.p(f"- Operaciones con sobrestadía>0: {con.sum()} "
            f"(días depósito mediana con sobrestadía={d.loc[con,'dias_deposito'].median():.0f} "
            f"vs sin={d.loc[~con,'dias_deposito'].median():.0f})")

    # ---------- estacionalidad / tendencia ----------
    R.h("Tendencia temporal del costo (inflación de tarifas)")
    by_year = df.groupby("anio").agg(
        n=(TGT, "size"), serv_mediana=(TGT, "median"),
        unit_valor_med=("costo_unit_valor", "median"),
        flete_med=("FLETE_INTERNACIONAL", "median") if "FLETE_INTERNACIONAL" in df else (TGT, "median"),
    )
    R.p(by_year.round(3).to_markdown())
    # tendencia del costo unitario (neto de volumen): regresión sobre año
    du = df[["anio", "costo_unit_valor"]].dropna()
    du = du[(du["costo_unit_valor"] > 0) & (du["costo_unit_valor"] < du["costo_unit_valor"].quantile(.99))]
    sl, ic, r, p, se = stats.linregress(du["anio"].astype(float), np.log(du["costo_unit_valor"]))
    R.p(f"\n- Tendencia log(costo/valor) ~ año: pendiente={sl:+.3f}/año "
        f"(≈{100*(np.exp(sl)-1):+.1f}%/año), p={p:.1e}. "
        f"{'Hay' if p<0.05 else 'No hay'} deriva temporal significativa ⇒ "
        f"**validación temporal (forward chaining), no K-fold aleatorio.**")

    by_month = df.groupby("mes_arribo")[TGT].median()

    # ---------- figura ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    # tiempos boxplot
    td = df[tcols].melt(var_name="tiempo", value_name="dias").dropna()
    td = td[(td["dias"] >= 0) & (td["dias"] < 200)]
    sns.boxplot(data=td, x="tiempo", y="dias", ax=ax[0], showfliers=False)
    ax[0].tick_params(axis="x", rotation=30, labelsize=8); ax[0].set_title("Tiempos del proceso")
    # tendencia anual
    by_year["serv_mediana"].plot(marker="o", ax=ax[1], color="#2b7bba")
    ax[1].set_title("Servicios (mediana) por año de arribo"); ax[1].set_ylabel("USD")
    ax2 = ax[1].twinx(); by_year["unit_valor_med"].plot(marker="s", ax=ax2, color="#c44e52")
    ax2.set_ylabel("costo/valor", color="#c44e52")
    # estacionalidad mensual
    by_month.plot(kind="bar", ax=ax[2], color="#3aa17e")
    ax[2].set_title("Servicios (mediana) por mes de arribo"); ax[2].set_xlabel("mes")
    fig.suptitle("4.5 — Tiempos y estacionalidad", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "f06_temporal.png")); plt.close(fig)

    # ---------- catálogo de derivadas ----------
    R.h("Variables derivadas propuestas (con justificación de negocio)")
    derivadas = [
        ("transit_days", "ata − atd", "Tránsito largo (marítimo) ⇒ más flete y riesgo de sobrestadía."),
        ("dias_deposito", "retiro_t1 − descarga", "Días en depósito temporal ⇒ almacenaje/sobrestadía."),
        ("es_aereo", "mode == AIR", "Separa la estructura de costo aérea vs marítima."),
        ("incoterm_grupo", "incoterm → quién paga flete", "EXW/FOB: importador asume flete (más costo logístico)."),
        ("categoria_canon", "categoría armonizada", "Colapsa la deriva semántica de CATEGORY a ~8 grupos."),
        ("costo_unit_valor", "servicios / valor_mercadería", "Ratio más estable; normaliza por tamaño de la operación."),
        ("ratio_flete_valor", "flete / valor", "Intensidad logística relativa al valor."),
        ("canal_rojo", "canal == rojo", "Aforo físico ⇒ más inspección/tiempo/costo."),
        ("requiere_senasa", "tiene fecha SENASA", "Plantas/sustratos ⇒ trámite fitosanitario extra."),
        ("tiene_seguro", "insurance_hper == SI", "Operaciones aseguradas (perfil de riesgo/valor)."),
        ("mes_arribo / anio", "de ata", "Estacionalidad e inflación de tarifas."),
        ("zona_ruta", "POL→POD agrupado", "Corredor logístico (Chile-Callao, Europa-Callao, USA-aéreo)."),
    ]
    R.p(pd.DataFrame(derivadas, columns=["variable", "fórmula", "justificación"]).to_markdown(index=False))

    R.save(os.path.join(C.REPORTS_DIR, "08_temporal.md"))
    print("OK 4.5 -> 08_temporal.md ; pendiente_temporal:", f"{sl:+.3f}/año p={p:.1e}")


if __name__ == "__main__":
    main()
