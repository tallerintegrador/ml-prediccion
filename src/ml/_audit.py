# -*- coding: utf-8 -*-
"""Auditoría técnica rápida previa al modelado. No modifica nada."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import config as C
import dataset as D

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

print("=" * 70)
print("1) dataset.load()")
print("=" * 70)
full = D.load(fiable_only=False)
fiab = D.load(fiable_only=True)
print("full (todas las joineadas):", full.shape)
print("fiable_only:", fiab.shape)
print("target_fiable value_counts:\n", full["target_fiable"].value_counts(dropna=False))

print("\n-- targets (fiable) --")
print(fiab[["target_total", "tributos_usd", "target_servicios"]].describe().round(1))
print("servicios <=0:", int((fiab["target_servicios"] <= 0).sum()))
print("skew servicios:", round(float(fiab["target_servicios"].skew()), 2),
      "| skew log:", round(float(fiab["log_target_servicios"].skew()), 2))

print("\n-- despachadas sin facturar (target no fiable) --")
nofi = full[~full["target_fiable"].astype(bool)]
print("n:", nofi.shape[0])
print("status:\n", nofi["status"].value_counts(dropna=False).head(15))

print("\n" + "=" * 70)
print("2) Leakage: ¿están las prohibidas en FEATURES?")
print("=" * 70)
banned = ["fecha_liquidacion", "fecha_registro_gastos", "tributos_usd",
          "igv_usd", "target_usd", "target_usd_sin_trib", "n_conceptos",
          "n_lineas_costo", "status"]
feats = D.FEATURES_NUM + D.FEATURES_CAT
print("FEATURES_NUM:", D.FEATURES_NUM)
print("FEATURES_CAT:", D.FEATURES_CAT)
print("Prohibidas presentes en features:", [b for b in banned if b in feats])

print("\n" + "=" * 70)
print("3) Disponibilidad al predecir: ¿features pobladas en NO fiables?")
print("=" * 70)
avail = []
for c in feats:
    if c in nofi:
        avail.append((c, round(100 * nofi[c].notna().mean(), 1),
                      round(100 * fiab[c].notna().mean(), 1)))
    else:
        avail.append((c, None, None))
av = pd.DataFrame(avail, columns=["feature", "pct_notnull_pend", "pct_notnull_fiable"])
print(av.to_string(index=False))

print("\n" + "=" * 70)
print("4) Nulos / cardinalidad / outliers en FIABLE")
print("=" * 70)
diag = []
for c in feats:
    if c not in fiab:
        diag.append((c, "FALTA", None, None)); continue
    s = fiab[c]
    diag.append((c, round(100 * s.isna().mean(), 1), s.nunique(dropna=True),
                 str(s.dtype)))
dg = pd.DataFrame(diag, columns=["feature", "pct_nulos", "cardinalidad", "dtype"])
print(dg.to_string(index=False))

print("\n-- numéricas: rango/outliers --")
print(fiab[D.FEATURES_NUM].describe(percentiles=[.5, .95, .99]).round(1).T)

print("\n" + "=" * 70)
print("5) Segmento mode x categoria_canon (para baseline)")
print("=" * 70)
seg = (fiab.groupby(["mode", "categoria_canon"], observed=True)["target_servicios"]
       .agg(["size", "median", "mean"]).round(0).sort_values("size", ascending=False))
print(seg.head(20))
print("n segmentos:", seg.shape[0])

print("\n-- distribución temporal (anio) --")
print(fiab["anio"].value_counts().sort_index())

print("\n" + "=" * 70)
print("Parquets auxiliares")
print("=" * 70)
for f in ["target_por_concepto.parquet", "expense_lineas.parquet", "operativo_lineas.parquet"]:
    p = os.path.join(C.PROC_DIR, f)
    df = pd.read_parquet(p)
    print(f"\n{f}: {df.shape}")
    print("cols:", list(df.columns)[:30])
