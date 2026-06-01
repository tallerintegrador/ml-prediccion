"""
04c_clustering.py — CRISP-DM Fase 4 (Track C): Segmentación de despachos.

Metodología:
  1. Agregación a nivel de operación (1 fila = 1 despacho completo)
  2. Preprocesamiento: StandardScaler + OneHotEncoder → ColumnTransformer
  3. Reducción dimensional: PCA (≥90% varianza)
  4. K-Means: búsqueda del K óptimo por silhouette + elbow (K=3..9)
  5. HDBSCAN: ajuste de hiperparámetros por silhouette, identificación de outliers
  6. Caracterización de clusters con perfil descriptivo y nombre de negocio
  7. PSI (Population Stability Index): detección de drift distribucional train→test

Prerequisito: python src/03_feature_engineering.py
Ejecutar    : python src/04c_clustering.py
Salidas     : models/kmeans_k{K}.joblib
              models/hdbscan_model.joblib
              models/clustering_preprocessor.joblib
              models/clustering_pca.joblib
              models/psi_baseline.joblib
              models/cluster_nombres.joblib
              data/processed/operaciones_con_clusters.parquet
              reports/perfil_clusters.csv
              reports/psi_baseline.csv
              reports/figures/clustering_*.png
"""

import sys
import warnings
from pathlib import Path

import hdbscan
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    RANDOM_STATE, DPI_FIGURAS, COLOR_PRIMARIO, PALETTE
)
from utils import calcular_psi_numerico, calcular_psi_categorico, interpretar_psi

# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_test.parquet")

print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

# ---------------------------------------------------------------------------
# 1. Agregación a nivel de operación
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("1. AGREGACIÓN POR OPERACIÓN")
print("=" * 65)

def modo_mas_frecuente(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) > 0 else "DESCONOCIDO"

def agregar_por_operacion(df):
    agg = {"importe_total_pen": "sum", "concepto_canonico": "nunique"}
    cols_agg = {
        "n_facturas":       ("importe_total_pen", "count"),
        "n_conceptos":      ("concepto_canonico", "nunique"),
        "costo_total_pen":  ("importe_total_pen", "sum"),
        "costo_mediano":    ("importe_total_pen", "median"),
    }
    if "proveedor" in df.columns:
        cols_agg["n_proveedores"] = ("proveedor", "nunique")
    elif "proveedor_norm" in df.columns:
        cols_agg["n_proveedores"] = ("proveedor_norm", "nunique")
    if "peso_bruto" in df.columns:
        cols_agg["peso_total_kg"]      = ("peso_bruto", "max")
    if "contenedores" in df.columns:
        cols_agg["total_contenedores"] = ("contenedores", "max")
    if "bultos" in df.columns:
        cols_agg["total_bultos"]       = ("bultos", "max")
    if "dias_transito" in df.columns:
        cols_agg["dias_transito_prom"] = ("dias_transito", "mean")

    ops = df.groupby("nro_operacion").agg(**cols_agg).reset_index()

    for col in ["mode", "incoterm_grupo", "ruta_origen", "pais_origen", "agencia_aduana"]:
        if col in df.columns:
            moda   = df.groupby("nro_operacion")[col].agg(modo_mas_frecuente)
            ops[col] = ops["nro_operacion"].map(moda).fillna("DESCONOCIDO")

    if "campaña" in df.columns:
        ops["campaña"] = df.groupby("nro_operacion")["campaña"].first().reindex(ops["nro_operacion"]).values

    return ops

ops_train = agregar_por_operacion(train)
ops_test  = agregar_por_operacion(test)

print(f"  Operaciones train: {len(ops_train):,}")
print(f"  Operaciones test : {len(ops_test):,}")

# ---------------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. PREPROCESAMIENTO (StandardScaler + OneHotEncoder)")
print("=" * 65)

NUM_COLS = [c for c in [
    "costo_total_pen", "n_facturas", "n_conceptos", "n_proveedores",
    "costo_mediano", "peso_total_kg", "total_contenedores",
    "total_bultos", "dias_transito_prom",
] if c in ops_train.columns]

CAT_COLS = [c for c in [
    "mode", "incoterm_grupo", "ruta_origen", "agencia_aduana",
] if c in ops_train.columns]

print(f"  Numéricas  ({len(NUM_COLS)}): {NUM_COLS}")
print(f"  Categóricas ({len(CAT_COLS)}): {CAT_COLS}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
], remainder="drop")

fill_num = ops_train[NUM_COLS].median()
fill_cat = {c: ops_train[c].mode().iloc[0] if len(ops_train[c].dropna()) > 0 else "DESCONOCIDO"
            for c in CAT_COLS}

def rellenar(df_ops):
    df2 = df_ops.copy()
    df2[NUM_COLS] = df2[NUM_COLS].fillna(fill_num)
    for c, v in fill_cat.items():
        if c in df2.columns:
            df2[c] = df2[c].fillna(v)
    return df2[NUM_COLS + CAT_COLS]

X_train_prep = preprocessor.fit_transform(rellenar(ops_train))
X_test_prep  = preprocessor.transform(rellenar(ops_test))

print(f"  Dimensión tras preprocesamiento: train={X_train_prep.shape}, test={X_test_prep.shape}")

# ---------------------------------------------------------------------------
# 3. PCA
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. PCA (≥90% varianza explicada)")
print("=" * 65)

N_MAX  = min(15, X_train_prep.shape[1], X_train_prep.shape[0] - 1)
pca    = PCA(n_components=N_MAX, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_prep)
X_test_pca  = pca.transform(X_test_prep)

var_acum = np.cumsum(pca.explained_variance_ratio_)
n_comps  = int(np.searchsorted(var_acum, 0.90)) + 1
print(f"  Varianza acumulada con {N_MAX} componentes: {var_acum[-1]*100:.1f}%")
print(f"  Componentes para ≥90% varianza: {n_comps}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, N_MAX + 1), pca.explained_variance_ratio_ * 100, color=COLOR_PRIMARIO, label="Individual")
ax.plot(range(1, N_MAX + 1), var_acum * 100, "r-o", markersize=4, label="Acumulada")
ax.axhline(90, color="orange", linestyle="--", linewidth=1, label="90% objetivo")
ax.set_xlabel("Componente principal", fontsize=11)
ax.set_ylabel("Varianza explicada (%)", fontsize=11)
ax.set_title("Scree plot — PCA de operaciones de importación", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_01_pca_scree.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: clustering_01_pca_scree.png")

# ---------------------------------------------------------------------------
# 4. K-Means: búsqueda del K óptimo
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("4. K-MEANS — BÚSQUEDA DEL K ÓPTIMO (K=3..9)")
print("=" * 65)

K_RANGE     = range(3, 10)
inertias    = []
silhouettes = []

for k in K_RANGE:
    km_k  = KMeans(n_clusters=k, n_init=10, max_iter=200, random_state=RANDOM_STATE)
    lbl_k = km_k.fit_predict(X_train_pca)
    inertias.append(km_k.inertia_)
    sil_k = silhouette_score(X_train_pca, lbl_k, sample_size=min(3000, len(X_train_pca)))
    silhouettes.append(sil_k)
    print(f"  K={k}: inertia={km_k.inertia_:,.0f}, silhouette={sil_k:.3f}")

K_OPTIMO = list(K_RANGE)[int(np.argmax(silhouettes))]
print(f"\n  → K óptimo por silhouette máximo: K={K_OPTIMO}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(K_RANGE), inertias, "bo-", linewidth=2)
axes[0].axvline(K_OPTIMO, color="red", linestyle="--", linewidth=1, label=f"K={K_OPTIMO}")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow plot — K-Means", fontsize=12, fontweight="bold"); axes[0].legend()

axes[1].plot(list(K_RANGE), silhouettes, "go-", linewidth=2)
axes[1].axvline(K_OPTIMO, color="red", linestyle="--", linewidth=1, label=f"K={K_OPTIMO}")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette por K", fontsize=12, fontweight="bold"); axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_00_k_search.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: clustering_00_k_search.png")

print(f"\n=== K-MEANS (K={K_OPTIMO}) ===")
kmeans = KMeans(n_clusters=K_OPTIMO, n_init=20, max_iter=300, random_state=RANDOM_STATE)
kmeans.fit(X_train_pca)

labels_km_train = kmeans.labels_
labels_km_test  = kmeans.predict(X_test_pca)

sil_km = silhouette_score(X_train_pca, labels_km_train, sample_size=min(5000, len(X_train_pca)))
db_km  = davies_bouldin_score(X_train_pca, labels_km_train)
print(f"  Silhouette score: {sil_km:.3f}  (mayor = mejor)")
print(f"  Davies-Bouldin  : {db_km:.3f}  (menor = mejor)")

dist_km = pd.Series(labels_km_train).value_counts().sort_index()
print(f"  Distribución de clusters:\n{dist_km.to_string()}")

ops_train["cluster_kmeans"] = labels_km_train
ops_test["cluster_kmeans"]  = labels_km_test

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=labels_km_train,
                cmap="tab10", alpha=0.5, s=20)
ax.set_xlabel("PC1", fontsize=11); ax.set_ylabel("PC2", fontsize=11)
ax.set_title(f"K-Means K={K_OPTIMO} — Proyección PC1 vs PC2 (silhouette={sil_km:.2f})",
             fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_02_kmeans_pca.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: clustering_02_kmeans_pca.png")

# ---------------------------------------------------------------------------
# 5. HDBSCAN
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. HDBSCAN (búsqueda de hiperparámetros)")
print("=" * 65)

mejores_hdb = {"sil": -1.0, "min_cluster_size": 20, "min_samples": 5}

for mcs in [10, 20, 30, 40]:
    for ms in [3, 5, 8]:
        try:
            h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                 metric="euclidean", core_dist_n_jobs=-1)
            lbl = h.fit_predict(X_train_pca)
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            if n_cl < 2: continue
            mask_no = lbl != -1
            if mask_no.sum() < 50: continue
            sil = silhouette_score(X_train_pca[mask_no], lbl[mask_no],
                                   sample_size=min(3000, mask_no.sum()))
            pct_out = (lbl == -1).mean() * 100
            print(f"  mcs={mcs:>2}, ms={ms}: clusters={n_cl}, sil={sil:.3f}, outliers={pct_out:.1f}%")
            if sil > mejores_hdb["sil"]:
                mejores_hdb = {"sil": sil, "min_cluster_size": mcs, "min_samples": ms}
        except Exception:
            continue

print(f"\n  → Mejores params: mcs={mejores_hdb['min_cluster_size']}, ms={mejores_hdb['min_samples']}  (sil={mejores_hdb['sil']:.3f})")

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=mejores_hdb["min_cluster_size"],
    min_samples=mejores_hdb["min_samples"],
    metric="euclidean", prediction_data=True, core_dist_n_jobs=-1,
)
labels_hdb_train = hdbscan_model.fit_predict(X_train_pca)
labels_hdb_test, _ = hdbscan.approximate_predict(hdbscan_model, X_test_pca)

n_cl_hdb    = len(set(labels_hdb_train)) - (1 if -1 in labels_hdb_train else 0)
pct_out_hdb = (labels_hdb_train == -1).mean() * 100
print(f"  HDBSCAN final: {n_cl_hdb} clusters, {pct_out_hdb:.1f}% outliers")

mask_hdb  = labels_hdb_train != -1
sil_hdb   = (silhouette_score(X_train_pca[mask_hdb], labels_hdb_train[mask_hdb],
                               sample_size=min(5000, mask_hdb.sum()))
             if mask_hdb.sum() > 50 else 0.0)
db_hdb    = (davies_bouldin_score(X_train_pca[mask_hdb], labels_hdb_train[mask_hdb])
             if mask_hdb.sum() > 50 else np.nan)

print(f"  Silhouette HDBSCAN (sin outliers): {sil_hdb:.3f}")
print(f"  Davies-Bouldin HDBSCAN           : {db_hdb:.3f}")

ops_train["cluster_hdbscan"] = labels_hdb_train
ops_test["cluster_hdbscan"]  = labels_hdb_test

fig, ax = plt.subplots(figsize=(9, 6))
n_colors = max(n_cl_hdb, 1)
cmap_hdb = plt.colormaps["tab10"].resampled(n_colors + 1)
cols_hdb = [cmap_hdb(int(l)) if l != -1 else (0.5, 0.5, 0.5, 0.3) for l in labels_hdb_train]
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cols_hdb, alpha=0.5, s=20)
ax.set_xlabel("PC1", fontsize=11); ax.set_ylabel("PC2", fontsize=11)
ax.set_title(f"HDBSCAN — {n_cl_hdb} clusters + outliers (gris, {pct_out_hdb:.1f}%)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_03_hdbscan_pca.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: clustering_03_hdbscan_pca.png")

# ---------------------------------------------------------------------------
# 6. Caracterización y nombres de negocio de los clusters K-Means
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("6. PERFIL Y NOMBRES DE NEGOCIO DE CLUSTERS")
print("=" * 65)

def asignar_nombre_cluster(perfil: dict) -> str:
    """Asigna nombre descriptivo de negocio basado en el perfil del cluster."""
    modo   = str(perfil.get("modo_dominante",  "FCL")).upper()
    ruta   = str(perfil.get("ruta_dominante",  "OTROS")).upper()
    costo  = perfil.get("costo_mediano_kpen", 0)
    n_fact = perfil.get("n_facturas_median",  0)
    ctnrs  = perfil.get("contenedores_median", 0)
    n_conc = perfil.get("n_conceptos_median", 0)

    if "AIR" in modo:
        if costo < 20:
            return "Carga Aérea Ligera / Urgente"
        return "Carga Aérea de Alto Valor"
    if "LCL" in modo:
        return "Consolidados LCL — Pequeño Volumen"
    if costo > 300:
        return f"Despacho FCL Masivo — {ruta.title()}"
    if ctnrs >= 3:
        return f"Multi-Contenedor Estándar — {ruta.title()}"
    if n_conc <= 3:
        return "Operación Simple — Pocos Conceptos"
    if "EUROPA" in ruta:
        return "Importación Europea FCL Estándar"
    if "LATAM" in ruta:
        return "Proveedor Regional LATAM"
    if "ASIA" in ruta:
        return "Importación Asiática"
    if "NORTEAM" in ruta:
        return "Importación Norteamérica"
    return f"Despacho Mixto — {ruta.title()}"

perfil_clusters = []
cluster_nombres = {}

for k in range(K_OPTIMO):
    mask = ops_train["cluster_kmeans"] == k
    sub  = ops_train[mask]
    if len(sub) == 0:
        continue
    row = {
        "cluster_kmeans":       k,
        "n_operaciones":        len(sub),
        "costo_mediano_kpen":   round(sub["costo_total_pen"].median() / 1_000, 1),
        "n_facturas_median":    round(sub["n_facturas"].median(), 1),
        "n_conceptos_median":   round(sub["n_conceptos"].median(), 1),
    }
    if "mode" in sub.columns:
        row["modo_dominante"] = sub["mode"].value_counts().index[0] if len(sub) > 0 else "-"
    if "ruta_origen" in sub.columns:
        row["ruta_dominante"] = sub["ruta_origen"].value_counts().index[0] if len(sub) > 0 else "-"
    if "total_contenedores" in sub.columns:
        row["contenedores_median"] = round(sub["total_contenedores"].median(), 1)

    nombre = asignar_nombre_cluster(row)
    row["nombre_negocio"] = nombre
    cluster_nombres[k]    = nombre
    perfil_clusters.append(row)

df_perfil = pd.DataFrame(perfil_clusters)
print(df_perfil[["cluster_kmeans", "n_operaciones", "costo_mediano_kpen",
                  "n_facturas_median", "modo_dominante" if "modo_dominante" in df_perfil.columns else "n_conceptos_median",
                  "nombre_negocio"]].to_string(index=False))

df_perfil.to_csv(REPORTS_DIR / "perfil_clusters.csv", index=False)
print("\nGuardado: reports/perfil_clusters.csv")

ops_train["nombre_cluster"] = ops_train["cluster_kmeans"].map(cluster_nombres)
ops_test["nombre_cluster"]  = ops_test["cluster_kmeans"].map(cluster_nombres)

# ---------------------------------------------------------------------------
# 7. PSI — detección de drift train → test
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. PSI — DETECCIÓN DE DRIFT DISTRIBUCIONAL")
print("=" * 65)

num_psi = [c for c in ["costo_total_pen", "n_facturas", "n_conceptos",
                         "peso_total_kg", "dias_transito_prom"] if c in ops_train.columns]
cat_psi = [c for c in ["mode", "incoterm_grupo", "ruta_origen"] if c in ops_train.columns]

psi_resultados = []
for col in num_psi:
    psi_v = calcular_psi_numerico(ops_train[col].dropna(), ops_test[col].dropna())
    psi_resultados.append({"feature": col, "tipo": "numerica",
                            "PSI": round(psi_v, 4) if not pd.isna(psi_v) else np.nan,
                            "estado": interpretar_psi(psi_v)})
for col in cat_psi:
    psi_v = calcular_psi_categorico(ops_train[col], ops_test[col])
    psi_resultados.append({"feature": col, "tipo": "categorica",
                            "PSI": round(psi_v, 4),
                            "estado": interpretar_psi(psi_v)})

df_psi = pd.DataFrame(psi_resultados).sort_values("PSI", ascending=False)
print(f"\nResultados PSI (train 2020-2024 → test 2025+):")
print(df_psi.to_string(index=False))
df_psi.to_csv(REPORTS_DIR / "psi_baseline.csv", index=False)
print("\nGuardado: reports/psi_baseline.csv")

colores_map = {"ESTABLE": "mediumseagreen", "CAMBIO MODERADO": "orange",
               "DRIFT CRITICO": "red", "SIN DATOS": "gray"}
fig, ax = plt.subplots(figsize=(9, 5))
colores_bar = [colores_map.get(e, "gray") for e in df_psi["estado"]]
ax.barh(df_psi["feature"], df_psi["PSI"].fillna(0), color=colores_bar, edgecolor="white")
ax.axvline(0.10, color="orange", linestyle="--", linewidth=1.2, label="Moderado (PSI=0.10)")
ax.axvline(0.25, color="red",    linestyle="--", linewidth=1.2, label="Crítico  (PSI=0.25)")
ax.set_xlabel("PSI", fontsize=11)
ax.set_title("PSI — Drift distribucional train (2020-2024) vs test (2025+)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_04_psi_drift.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: clustering_04_psi_drift.png")

# ---------------------------------------------------------------------------
# Guardar artefactos
# ---------------------------------------------------------------------------
all_ops = pd.concat([
    ops_train.assign(split="train"),
    ops_test.assign(split="test"),
], ignore_index=True)

all_ops.to_parquet(PROCESSED_DIR / "operaciones_con_clusters.parquet", index=False)

psi_baseline = {
    "resultados":    df_psi,
    "train_period":  "2020-2024",
    "test_period":   "2025+",
    "fecha_calculo": pd.Timestamp.now().isoformat(),
}

nombre_archivo_kmeans = f"kmeans_k{K_OPTIMO}.joblib"
joblib.dump(kmeans,         MODELS_DIR / nombre_archivo_kmeans)
joblib.dump(hdbscan_model,  MODELS_DIR / "hdbscan_model.joblib")
joblib.dump(preprocessor,   MODELS_DIR / "clustering_preprocessor.joblib")
joblib.dump(pca,            MODELS_DIR / "clustering_pca.joblib")
joblib.dump(psi_baseline,   MODELS_DIR / "psi_baseline.joblib")
joblib.dump(cluster_nombres, MODELS_DIR / "cluster_nombres.joblib")
joblib.dump({
    "k_optimo":       K_OPTIMO,
    "sil_kmeans":     float(sil_km),
    "db_kmeans":      float(db_km),
    "sil_hdbscan":    float(sil_hdb),
    "num_cols":       NUM_COLS,
    "cat_cols":       CAT_COLS,
    "fill_num":       fill_num.to_dict(),
    "fill_cat":       fill_cat,
}, MODELS_DIR / "clustering_config.joblib")

print(f"\nGuardado: {PROCESSED_DIR / 'operaciones_con_clusters.parquet'}")
print(f"Modelos guardados en {MODELS_DIR}:")
for f in sorted(MODELS_DIR.glob("clustering*.joblib")) + sorted(MODELS_DIR.glob("kmeans*.joblib")) + sorted(MODELS_DIR.glob("hdbscan*.joblib")) + sorted(MODELS_DIR.glob("psi*.joblib")) + sorted(MODELS_DIR.glob("cluster_nombres*.joblib")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")

print("\nClustering finalizado. Próximo paso: python src/05_demo.py")
