# -*- coding: utf-8 -*-
"""
02_target.py — Conceptos canónicos, reconstrucción del target y join (4.0.5 / 4.1).

  - Canonicaliza los 81 conceptos de costo crudos -> 14 conceptos canónicos.
  - Reconstruye el TARGET = costo logístico total en USD por operación
        target_usd            = Σ Monto Total USD (excluye Percepción IGV)
        target_usd_sin_trib   = idem, excluyendo además DERECHOS_IMPUESTOS
  - Agrega los operativos a 1 fila por operación.
  - Une operativo + target por op_key (YY-NNN). Reporta cobertura, granularidad,
    duplicados y colisiones de llave.
  - Guarda data_csv/processed/operaciones_modelables.parquet y target_por_concepto.parquet
  - Figuras: distribución del target, composición por concepto, diagnóstico de join.
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

# columnas operativas: cómo agregar de línea -> operación
SUM_COLS = ["qty", "amount", "amount_usd", "bulks"]            # aditivas por línea-producto
OPLEVEL_NUM = ["ctnr_qty", "peso_bruto", "payment_term"]        # nivel-operación (no sumar)
FIRST_COLS = ["campania", "status", "importador", "buyer", "supplier", "proyecto",
              "categoria", "producto", "um", "moneda", "incoterm", "mode", "type",
              "country_origin", "pol", "pod", "ctnr_type", "bulks_type", "ffw",
              "ffw_reference", "shipping_line", "vessel", "depot", "customs_agent",
              "canal", "insurance_hper", "delivery_type", "punto_llegada",
              "transp_t1", "transp_t2", "schema", "source_file", "anio_op"]
DATE_COLS = ["cargo_ready", "quote_request_date", "assignment_date", "pick_up_date",
             "atd", "ata", "ie_date", "fecha_num_dam", "fecha_descarga", "fecha_tarja",
             "fecha_levante", "senasa_inspeccion", "senasa_liberacion", "demurrage_exp",
             "fecha_retiro_t1", "fecha_entrega_base_t2", "fecha_fin_viaje",
             "receipt_confirmation", "fecha_liquidacion", "fecha_registro_gastos"]

STATUS_FIABLE = {"liquidada", "liquidado", "entregado", "entregada", "archivada",
                 "archivado", "cerrada", "cerrado"}


def first_valid(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


# Llave canónica de operación: op_id_full (YYNNN+CÓDIGO) -> 0 colisiones.
# op_key (YY-NNN) se conserva sólo como diagnóstico.
KEY = "op_id_full"


def aggregate_operative(op):
    op = op[op[KEY].notna()].copy()
    agg = {}
    for c in SUM_COLS:
        if c in op.columns:
            agg[c] = (c, "sum")
    for c in OPLEVEL_NUM + DATE_COLS + FIRST_COLS + ["op_key"]:
        if c in op.columns:
            agg[c] = (c, first_valid)
    g = op.groupby(KEY).agg(**agg)
    g["n_oc_lineas"] = op.groupby(KEY).size()
    g["n_oc_distintas"] = op.groupby(KEY)["oc"].nunique()
    g["n_productos"] = op.groupby(KEY)["producto"].nunique()
    return g.reset_index()


def build_target(exp):
    exp = exp[exp[KEY].notna()].copy()
    # recomputar canónico por si cambiaron las reglas
    exp["concepto_canon"] = exp["concepto_raw"].map(U.canon_concept)
    logist = exp[~exp["concepto_canon"].isin(C.CONCEPT_EXCLUDE_TARGET)]
    tgt = logist.groupby(KEY).agg(
        target_usd=("monto_usd", "sum"),
        n_conceptos=("concepto_canon", "nunique"),
        n_lineas_costo=("monto_usd", "size"),
        campania_exp=("campania_exp", first_valid),
    ).reset_index()
    sin_trib = logist[~logist["concepto_canon"].isin(C.CONCEPT_TAXES)]
    tgt = tgt.merge(
        sin_trib.groupby(KEY)["monto_usd"].sum().rename("target_usd_sin_trib"),
        on=KEY, how="left")
    igv = exp[exp["concepto_canon"] == "PERCEPCION_IGV"].groupby(KEY)["monto_usd"].sum()
    tgt = tgt.merge(igv.rename("igv_usd"), on=KEY, how="left")
    # tributos (derechos/Ad Valorem) como componente separado -> se modela aparte
    trib = (logist[logist["concepto_canon"].isin(C.CONCEPT_TAXES)]
            .groupby(KEY)["monto_usd"].sum())
    tgt = tgt.merge(trib.rename("tributos_usd"), on=KEY, how="left")
    tgt["tributos_usd"] = tgt["tributos_usd"].fillna(0.0)
    # pivot concepto x operacion (para modelado multi-concepto posterior)
    pivot = logist.pivot_table(index=KEY, columns="concepto_canon",
                               values="monto_usd", aggfunc="sum", fill_value=0.0)
    return tgt, pivot


def report_concepts(exp, R):
    exp = exp.copy()
    exp["concepto_canon"] = exp["concepto_raw"].map(U.canon_concept)
    g = exp.groupby("concepto_canon").agg(
        n_lineas=("concepto_canon", "size"),
        usd_total=("monto_usd", "sum"),
        n_originales=("concepto_raw", "nunique"),
    ).sort_values("usd_total", ascending=False)
    g["usd_pct"] = 100 * g["usd_total"] / g["usd_total"].sum()
    ejemplos = (exp.groupby("concepto_canon")["concepto_raw"]
                .agg(lambda s: ", ".join(s.value_counts().head(3).index)))
    g["ejemplos"] = ejemplos
    g = g.reset_index()
    R.h("Conceptos canónicos (81 crudos → 14 canónicos)")
    R.p("Reglas regex aplicadas en orden sobre el texto normalizado; primera que matchea gana.")
    R.p(g.assign(usd_total=g["usd_total"].round(0), usd_pct=g["usd_pct"].round(1))
        .to_markdown(index=False))
    # detección de duplicados por normalización
    exp["norm"] = exp["concepto_raw"].map(U.norm_text)
    dup = (exp.groupby("norm")["concepto_raw"].agg(lambda s: sorted(s.unique()))
           .loc[lambda s: s.map(len) > 1])
    R.h("Duplicados detectados por normalización (mayúsculas/acentos/espacios)")
    if len(dup):
        for norm, variants in dup.items():
            R.p(f"- `{norm}` ← {variants}")
    else:
        R.p("Ninguno.")
    return g


def make_figures(tgt, concept_g, joined):
    # Fig 1: distribución del target
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    t = tgt["target_usd"].dropna()
    t = t[t > 0]
    ax[0].hist(t, bins=60, color="#2b7bba")
    ax[0].set_title("Target (USD) — escala lineal"); ax[0].set_xlabel("Costo logístico USD")
    ax[1].hist(np.log10(t), bins=60, color="#2b7bba")
    ax[1].set_title("log10(Target)"); ax[1].set_xlabel("log10 USD")
    ax[2].boxplot(t, vert=True, showfliers=True)
    ax[2].set_yscale("log"); ax[2].set_title("Boxplot (log)"); ax[2].set_ylabel("USD")
    fig.suptitle("4.3 — Distribución del costo logístico por operación", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f02_target_dist.png")); plt.close(fig)

    # Fig 2: composición por concepto
    fig, ax = plt.subplots(figsize=(9, 5))
    d = concept_g.sort_values("usd_total")
    ax.barh(d["concepto_canon"], d["usd_total"] / 1e6, color="#3aa17e")
    for i, (v, p) in enumerate(zip(d["usd_total"] / 1e6, d["usd_pct"])):
        ax.text(v, i, f" {p:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("USD (millones)"); ax.set_title("Composición del gasto por concepto canónico")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f02_conceptos.png")); plt.close(fig)

    # Fig 3: diagnóstico join
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["Join exitoso\n(oper+target)", "Solo operativo\n(sin facturar)", "Solo expense\n(huérfano)"]
    vals = [joined["join_ok"], joined["solo_op"], joined["solo_exp"]]
    ax.bar(cats, vals, color=["#3aa17e", "#e0a458", "#c44e52"])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("nº operaciones"); ax.set_title("Diagnóstico de unión expense ↔ operativo")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f02_join.png")); plt.close(fig)


def main():
    op = pd.read_parquet(os.path.join(C.PROC_DIR, "operativo_lineas.parquet"))
    exp = pd.read_parquet(os.path.join(C.PROC_DIR, "expense_lineas.parquet"))

    R = U.MdReport("4.0.5 / 4.1 — Conceptos canónicos, target y join")

    concept_g = report_concepts(exp, R)
    tgt, pivot = build_target(exp)
    op_agg = aggregate_operative(op)

    # --- join (sobre op_id_full) ---
    op_keys = set(op_agg[KEY]); exp_keys = set(tgt[KEY])
    join_ok = len(op_keys & exp_keys)
    solo_op = len(op_keys - exp_keys)
    solo_exp = len(exp_keys - op_keys)
    joined_stats = dict(join_ok=join_ok, solo_op=solo_op, solo_exp=solo_exp)

    modelable = op_agg.merge(tgt, on=KEY, how="inner", suffixes=("", "_tgt"))
    modelable["target_fiable"] = modelable["status"].map(
        lambda s: U.norm_text(s) in STATUS_FIABLE)
    # diagnóstico de colisiones de la llave laxa YY-NNN (justifica usar op_id_full)
    coll_yy = (op[op["op_key"].notna()].groupby("op_key")["op_id_full"].nunique() > 1).sum()

    R.h("Reconstrucción del target")
    R.p("Llave de unión: **op_id_full** (identificador completo normalizado, p. ej. "
        "`20119HPER`). Elegida sobre la llave laxa `YY-NNN` porque ésta colapsaba "
        f"**{coll_yy}** operaciones distintas (mismo año+secuencia, distinto código de empresa).")
    R.p(f"- Operaciones únicas en EXPENSE: **{len(exp_keys):,}**")
    R.p(f"- Operaciones únicas en OPERATIVO: **{len(op_keys):,}**")
    R.p(f"- **Join exitoso (ambos): {join_ok:,}** "
        f"({100*join_ok/len(exp_keys):.1f}% del expense)")
    R.p(f"- Solo operativo (despachadas, sin gasto cargado aún): **{solo_op:,}**  "
        "← escenario real de predicción")
    R.p(f"- Solo expense (huérfanas, sin operativo — sobre todo formato legacy 2019): **{solo_exp:,}**")
    R.p(f"\n**Granularidad final:** 1 fila por operación. Dataset modelable = "
        f"**{modelable.shape[0]:,} operaciones × {modelable.shape[1]} columnas**.")
    R.p(f"- Promedio de conceptos por operación: {tgt['n_conceptos'].mean():.1f}")
    R.p(f"- Promedio de líneas de costo por operación: {tgt['n_lineas_costo'].mean():.1f}")
    R.p(f"- Colisiones con op_id_full: **0** (vs {coll_yy} con YY-NNN).")

    R.h("Estadísticos del target (USD)")
    desc = modelable[["target_usd", "target_usd_sin_trib", "igv_usd"]].describe(
        percentiles=[.05, .25, .5, .75, .95]).T
    desc["skew"] = [modelable[c].skew() for c in desc.index]
    desc["kurtosis"] = [modelable[c].kurtosis() for c in desc.index]
    R.p(desc.round(1).to_markdown())
    R.p(f"\n- Operaciones con target_usd > 0: "
        f"{(modelable['target_usd']>0).sum():,} / {len(modelable):,}")
    R.p(f"- Operaciones con target ≤ 0 o nulo: "
        f"{(~(modelable['target_usd']>0)).sum():,} (revisar)")

    R.h("Cobertura del target por estado (status)")
    cov = modelable.assign(tiene_target=modelable["target_usd"] > 0).groupby(
        modelable["status"].fillna("(nulo)")).agg(
        n=("op_key", "size"), con_target=("tiene_target", "sum"),
        target_medio=("target_usd", "median")).sort_values("n", ascending=False)
    R.p(cov.round(0).to_markdown())

    R.h("Cobertura del target por campaña")
    covc = modelable.groupby("campania_exp").agg(
        n=("op_key", "size"), target_medio=("target_usd", "median"),
        target_total=("target_usd", "sum")).round(0)
    R.p(covc.to_markdown())

    # guardar
    modelable.to_parquet(os.path.join(C.PROC_DIR, "operaciones_modelables.parquet"), index=False)
    pivot.reset_index().to_parquet(os.path.join(C.PROC_DIR, "target_por_concepto.parquet"), index=False)
    make_figures(tgt, concept_g, joined_stats)

    out = R.save(os.path.join(C.REPORTS_EDA, "03_conceptos_target_join.md"))
    print("modelable:", modelable.shape)
    print(f"join_ok={join_ok} solo_op={solo_op} solo_exp={solo_exp}")
    print("OK ->", out)


if __name__ == "__main__":
    main()
