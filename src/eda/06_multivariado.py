# -*- coding: utf-8 -*-
"""
06_multivariado.py — Análisis multivariado y de segmentos (sección 4.6).

  - Segmentos: costo de servicios por modo×categoría y país×incoterm (dónde se concentra el gasto).
  - Clustering no supervisado (KMeans sobre features estandarizadas) -> perfiles de operación.
  - Importancia preliminar de variables con LightGBM + permutation importance,
    con SPLIT TEMPORAL honesto (train años tempranos / test años tardíos).
    EXPLORATORIO: no es el modelo final. (SHAP no está instalado; se usa gain + permutación.)
Genera reports/09_multivariado.md y figuras f07_*.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import config as C
import utils as U
import dataset as D

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
TGT = "target_servicios"
NUM = ["amount_usd", "qty", "bulks", "ctnr_qty", "transit_days", "payment_term",
       "n_oc_distintas", "n_productos", "mes_arribo", "anio"]
CAT = ["mode", "type", "incoterm_grupo", "categoria_canon", "pais_origen",
       "customs_agent", "ffw", "shipping_line", "depot"]


def segment_tables(df, R):
    R.h("Segmentos: dónde se concentra el gasto de servicios")
    seg = df.pivot_table(index="categoria_canon", columns="mode", values=TGT,
                         aggfunc="median")
    cnt = df.pivot_table(index="categoria_canon", columns="mode", values=TGT, aggfunc="size")
    R.p("**Mediana de servicios (USD) por categoría × modo:**")
    R.p(seg.round(0).to_markdown())
    R.p("\n**Nº de operaciones por categoría × modo:**")
    R.p(cnt.fillna(0).astype(int).to_markdown())

    R.h("Gasto total de servicios por segmento (top 10)")
    g = (df.groupby(["categoria_canon", "mode"], observed=True)
         .agg(n=(TGT, "size"), total=(TGT, "sum"), mediana=(TGT, "median"))
         .sort_values("total", ascending=False).head(10))
    g["%_del_total"] = (100 * g["total"] / df[TGT].sum()).round(1)
    R.p(g.round(0).to_markdown())

    # país × incoterm_grupo
    pim = df.pivot_table(index="pais_origen", columns="incoterm_grupo", values=TGT,
                         aggfunc="median")
    R.h("Mediana de servicios por país × grupo de incoterm")
    R.p(pim.round(0).to_markdown())
    return seg


def clustering(df, R):
    R.h("Clustering no supervisado (KMeans) — perfiles de operación")
    feat = df.copy()
    X = pd.DataFrame({
        "log_valor": np.log1p(feat["amount_usd"].fillna(0)),
        "log_bulks": np.log1p(feat["bulks"].fillna(0)),
        "transit": feat["transit_days"].fillna(feat["transit_days"].median()),
        "es_aereo": feat["es_aereo"],
        "ctnr": feat["ctnr_qty"].fillna(0),
        "log_serv": np.log1p(feat[TGT].fillna(0)),
    })
    Xs = StandardScaler().fit_transform(X)
    best_k, best_s = None, -1
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=C.SEED, n_init=10).fit(Xs)
        s = silhouette_score(Xs, km.labels_)
        if s > best_s:
            best_k, best_s, best_km = k, s, km
    feat["cluster"] = best_km.labels_
    R.p(f"- k óptimo por silueta = **{best_k}** (silhouette={best_s:.2f})")
    prof = feat.groupby("cluster").agg(
        n=(TGT, "size"), serv_mediana=(TGT, "median"),
        valor_med=("amount_usd", "median"), bulks_med=("bulks", "median"),
        pct_aereo=("es_aereo", "mean"),
        categoria_top=("categoria_canon", lambda s: s.mode().iloc[0] if len(s.mode()) else "—"),
    )
    prof["pct_aereo"] = (100 * prof["pct_aereo"]).round(0)
    R.p(prof.round(0).to_markdown())

    pca = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(pca[:, 0], pca[:, 1], c=feat["cluster"], cmap="tab10", s=12, alpha=.6)
    ax.set_title(f"Clusters (KMeans k={best_k}) en espacio PCA"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f07_clusters.png")); plt.close(fig)
    return prof


def preliminary_importance(df, R):
    R.h("Importancia preliminar de variables (LightGBM, split temporal)")
    data = df.copy()
    for c in CAT:
        data[c] = data[c].astype("string").fillna("DESCONOCIDO").astype("category")
    y = np.log1p(data[TGT].fillna(0))
    X = data[NUM + CAT]
    # split temporal honesto
    tr = data["anio"] <= 2023
    te = data["anio"] >= 2024
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=C.SEED,
                              verbose=-1)
    model.fit(X[tr], y[tr], categorical_feature=CAT)
    pred = np.expm1(model.predict(X[te]))
    true = data.loc[te, TGT].values
    mask = true > 0
    mae = np.mean(np.abs(pred[mask] - true[mask]))
    mdape = np.median(np.abs(pred[mask] - true[mask]) / true[mask]) * 100
    wape = 100 * np.sum(np.abs(pred[mask] - true[mask])) / np.sum(true[mask])
    R.p(f"- Split: train (≤2023, n={tr.sum()}) → test (≥2024, n={te.sum()}).")
    R.p(f"- **MAE test = ${mae:,.0f}** | **MdAPE = {mdape:.1f}%** | **WAPE = {wape:.1f}%** "
        "(baseline exploratorio, sin tuning ni features derivadas completas).")

    imp = pd.DataFrame({"feature": X.columns, "gain": model.feature_importances_}
                       ).sort_values("gain", ascending=False)
    imp["gain_%"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
    perm = permutation_importance(model, X[te], y[te], n_repeats=8, random_state=C.SEED,
                                  scoring="neg_mean_absolute_error")
    imp = imp.merge(pd.DataFrame({"feature": X.columns, "perm": perm.importances_mean}),
                    on="feature")
    imp = imp.sort_values("perm", ascending=False)
    R.p("\n**Importancia (ganancia y permutación sobre test):**")
    R.p(imp[["feature", "gain_%", "perm"]].assign(perm=imp["perm"].round(3))
        .head(18).to_markdown(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    d = imp.sort_values("perm").tail(15)
    ax.barh(d["feature"], d["perm"], color="#2b7bba")
    ax.set_xlabel("permutation importance (Δ MAE en log)")
    ax.set_title("4.6 — Importancia preliminar (LightGBM, test ≥2024)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f07_importancia.png")); plt.close(fig)

    # heatmap segmentos
    seg = df.pivot_table(index="categoria_canon", columns="mode", values=TGT, aggfunc="median")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(seg, annot=True, fmt=".0f", cmap="rocket_r", ax=ax)
    ax.set_title("Mediana servicios USD: categoría × modo")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f07_segmentos.png")); plt.close(fig)
    return imp, (mae, mdape, wape)


def main():
    df = D.load(fiable_only=True)
    R = U.MdReport(f"4.6 — Análisis multivariado y de segmentos (n={len(df)})")
    segment_tables(df, R)
    clustering(df, R)
    imp, metr = preliminary_importance(df, R)
    R.save(os.path.join(C.REPORTS_EDA, "09_multivariado.md"))
    print("OK 4.6 -> 09_multivariado.md")
    print(f"baseline LightGBM test: MAE=${metr[0]:,.0f} MdAPE={metr[1]:.1f}% WAPE={metr[2]:.1f}%")
    print("top features:", imp["feature"].head(6).tolist())


if __name__ == "__main__":
    main()
