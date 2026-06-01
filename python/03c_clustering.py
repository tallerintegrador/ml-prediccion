"""
03c_clustering.py — Segmentación de operaciones de importación y monitoreo de drift.

Metodología: K-Means (K=6) + HDBSCAN sobre features agregadas a nivel de operación.
             PSI (Population Stability Index) para detección de drift de distribución.
Prerequisito: python python/02_feature_engineering.py
Ejecutar    : python python/03c_clustering.py
Salidas     : models/kmeans_k6.joblib
              models/hdbscan_model.joblib
              models/clustering_preprocessor.joblib
              models/clustering_pca15.joblib
              models/psi_baseline.joblib
              data/processed/operaciones_con_clusters.parquet
              reports/perfil_clusters.csv
              reports/psi_baseline_2024_vs_2025.csv
              reports/figures/clustering_*.png
"""

# --- celda 1 ---
# Imports
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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --- celda 2 ---
# Rutas y carga
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
for _d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_test.parquet")

print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

# --- celda 3 ---
# Agregación a nivel de operación (1 fila por nro_operacion)
# El clustering se hace sobre el despacho completo, no por factura individual
print("\n=== AGREGACIÓN POR OPERACIÓN ===")


def modo_mas_frecuente(s):
    """Valor más frecuente de una serie; devuelve 'DESCONOCIDO' si vacío."""
    s = s.dropna()
    return s.mode().iloc[0] if len(s) > 0 else "DESCONOCIDO"


def agregar_por_operacion(df):
    """Agrega el dataset al nivel de operación (una fila = un despacho)."""
    agg_dict = {
        "importe_total_pen":  ("importe_total_pen", "sum"),
        "n_facturas":         ("importe_total_pen", "count"),
        "n_conceptos":        ("concepto_canonico", "nunique"),
        "n_proveedores":      ("proveedor", "nunique") if "proveedor" in df.columns else ("concepto_canonico", "count"),
        "costo_mediano":      ("importe_total_pen", "median"),
    }
    if "peso_bruto" in df.columns:
        agg_dict["peso_total_kg"] = ("peso_bruto", "max")
    if "contenedores" in df.columns:
        agg_dict["total_contenedores"] = ("contenedores", "max")
    if "bultos" in df.columns:
        agg_dict["total_bultos"] = ("bultos", "max")
    if "dias_transito" in df.columns:
        agg_dict["dias_transito_prom"] = ("dias_transito", "mean")

    ops = df.groupby("nro_operacion").agg(**agg_dict).reset_index()

    # Columnas categóricas: tomar el valor más frecuente por operación
    for col in ["mode", "incoterm_grupo", "ruta_origen", "pais_origen", "agencia_aduana"]:
        if col in df.columns:
            moda = (
                df.groupby("nro_operacion")[col]
                .agg(lambda s: s.mode().iloc[0] if len(s.dropna()) > 0 else "DESCONOCIDO")
            )
            ops[col] = ops["nro_operacion"].map(moda).fillna("DESCONOCIDO")

    # Campaña (año)
    if "campaña" in df.columns:
        ops["campaña"] = df.groupby("nro_operacion")["campaña"].first().reindex(ops["nro_operacion"]).values

    return ops


ops_train = agregar_por_operacion(train)
ops_test  = agregar_por_operacion(test)

print(f"Operaciones train: {len(ops_train):,}")
print(f"Operaciones test : {len(ops_test):,}")
print(f"\nColumnas disponibles: {list(ops_train.columns)}")

# --- celda 4 ---
# Preprocesamiento: StandardScaler + OneHotEncoder + ColumnTransformer
print("\n=== PREPROCESAMIENTO ===")

NUM_COLS = [c for c in [
    "importe_total_pen", "n_facturas", "n_conceptos", "n_proveedores",
    "costo_mediano", "peso_total_kg", "total_contenedores",
    "total_bultos", "dias_transito_prom"
] if c in ops_train.columns]

CAT_COLS = [c for c in [
    "mode", "incoterm_grupo", "ruta_origen", "agencia_aduana"
] if c in ops_train.columns]

print(f"  Numéricas ({len(NUM_COLS)}): {NUM_COLS}")
print(f"  Categóricas ({len(CAT_COLS)}): {CAT_COLS}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ],
    remainder="drop",
)

num_fill = ops_train[NUM_COLS].median()
cat_fill = ops_train[CAT_COLS].mode().iloc[0]
fill_values = pd.concat([num_fill, cat_fill])

X_train_ops = ops_train[NUM_COLS + CAT_COLS].fillna(fill_values)
X_test_ops  = ops_test[NUM_COLS + CAT_COLS].fillna(fill_values)

X_train_prep = preprocessor.fit_transform(X_train_ops)
X_test_prep  = preprocessor.transform(X_test_ops)

print(f"  Dimensión tras preprocessing: {X_train_prep.shape}")

# --- celda 4b ---
# PCA: reducir a 15 componentes (captura ≥90% de varianza)
print("\n=== PCA ===")

N_COMPONENTS = min(15, X_train_prep.shape[1], X_train_prep.shape[0] - 1)
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train_prep)
X_test_pca  = pca.transform(X_test_prep)

varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
print(f"  Varianza acumulada con {N_COMPONENTS} componentes: {varianza_acumulada[-1]*100:.1f}%")

# Scree plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, N_COMPONENTS + 1), pca.explained_variance_ratio_ * 100, color="steelblue")
ax.plot(range(1, N_COMPONENTS + 1), varianza_acumulada * 100, "r-o", markersize=4)
ax.set_xlabel("Componente principal")
ax.set_ylabel("Varianza explicada (%)")
ax.set_title("Scree plot — PCA de operaciones")
ax.axhline(90, color="orange", linestyle="--", label="90% acumulado")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_01_pca_scree.png", dpi=120)
plt.close()
print("Guardado: clustering_01_pca_scree.png")

# --- celda 5 ---
# Búsqueda del K óptimo por silhouette + elbow, luego K-Means con ese K
print("\n=== BÚSQUEDA DEL K ÓPTIMO (K=3..9) ===")

K_RANGE    = range(3, 10)
inertias   = []
silhouettes = []

for k in K_RANGE:
    km_k  = KMeans(n_clusters=k, n_init=10, max_iter=200, random_state=42)
    lbl_k = km_k.fit_predict(X_train_pca)
    inertias.append(km_k.inertia_)
    sil_k = silhouette_score(X_train_pca, lbl_k,
                             sample_size=min(3000, len(X_train_pca)))
    silhouettes.append(sil_k)
    print(f"  K={k}: inertia={km_k.inertia_:,.0f}, silhouette={sil_k:.3f}")

K_OPTIMO = list(K_RANGE)[int(np.argmax(silhouettes))]
print(f"\n  K óptimo por silhouette máximo: K={K_OPTIMO}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, "bo-")
axes[0].axvline(K_OPTIMO, color="red", linestyle="--", label=f"K={K_OPTIMO}")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow plot — K-Means")
axes[0].legend()
axes[1].plot(list(K_RANGE), silhouettes, "go-")
axes[1].axvline(K_OPTIMO, color="red", linestyle="--", label=f"K={K_OPTIMO}")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette")
axes[1].set_title("Silhouette por K")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_00_k_search.png", dpi=120)
plt.close()
print("Guardado: clustering_00_k_search.png")

print(f"\n=== K-MEANS (K={K_OPTIMO}) ===")

kmeans = KMeans(n_clusters=K_OPTIMO, n_init=20, max_iter=300, random_state=42)
kmeans.fit(X_train_pca)

labels_km_train = kmeans.labels_
labels_km_test  = kmeans.predict(X_test_pca)

sil_km = silhouette_score(X_train_pca, labels_km_train,
                          sample_size=min(5000, len(X_train_pca)))
db_km  = davies_bouldin_score(X_train_pca, labels_km_train)
print(f"  Silhouette score (train): {sil_km:.3f}  (mayor es mejor)")
print(f"  Davies-Bouldin score    : {db_km:.3f}  (menor es mejor)")
print(f"  Distribución de clusters:\n"
      f"{pd.Series(labels_km_train).value_counts().sort_index().to_string()}")

ops_train["cluster_kmeans"] = labels_km_train
ops_test["cluster_kmeans"]  = labels_km_test

# Visualizar en espacio PC1-PC2
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1],
    c=labels_km_train, cmap="tab10", alpha=0.5, s=20
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"K-Means K=6 — Proyección PC1 vs PC2 (silhouette={sil_km:.2f})")
plt.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_02_kmeans_pca.png", dpi=120)
plt.close()
print("Guardado: clustering_02_kmeans_pca.png")

# --- celda 6 ---
# HDBSCAN: búsqueda de hiperparámetros
print("\n=== HDBSCAN (búsqueda de hiperparámetros) ===")

mejores_params = {"sil": -1, "min_cluster_size": 20, "min_samples": 5}

for mcs in [10, 20, 30, 40]:
    for ms in [3, 5, 8]:
        try:
            h = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                core_dist_n_jobs=-1,
            )
            lbl = h.fit_predict(X_train_pca)
            n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
            if n_clusters < 2:
                continue
            # Evaluar solo puntos no outlier
            mask_no_out = lbl != -1
            if mask_no_out.sum() < 50:
                continue
            sil = silhouette_score(X_train_pca[mask_no_out], lbl[mask_no_out],
                                   sample_size=min(3000, mask_no_out.sum()))
            pct_outliers = (lbl == -1).mean() * 100
            print(f"  mcs={mcs:>2}, ms={ms}: clusters={n_clusters}, sil={sil:.3f}, outliers={pct_outliers:.1f}%")
            if sil > mejores_params["sil"]:
                mejores_params = {"sil": sil, "min_cluster_size": mcs, "min_samples": ms}
        except Exception:
            continue

print(f"\n  Mejores params: min_cluster_size={mejores_params['min_cluster_size']}, "
      f"min_samples={mejores_params['min_samples']}  (sil={mejores_params['sil']:.3f})")

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=mejores_params["min_cluster_size"],
    min_samples=mejores_params["min_samples"],
    metric="euclidean",
    prediction_data=True,
    core_dist_n_jobs=-1,
)
labels_hdb_train = hdbscan_model.fit_predict(X_train_pca)

# Para test: usar approximate_predict
labels_hdb_test, _ = hdbscan.approximate_predict(hdbscan_model, X_test_pca)

n_clusters_hdb = len(set(labels_hdb_train)) - (1 if -1 in labels_hdb_train else 0)
pct_out_hdb = (labels_hdb_train == -1).mean() * 100
print(f"  HDBSCAN final: {n_clusters_hdb} clusters, {pct_out_hdb:.1f}% outliers")

mask_hdb = labels_hdb_train != -1
sil_hdb = silhouette_score(
    X_train_pca[mask_hdb], labels_hdb_train[mask_hdb],
    sample_size=min(5000, mask_hdb.sum())
) if mask_hdb.sum() > 50 else 0.0
db_hdb  = davies_bouldin_score(
    X_train_pca[mask_hdb], labels_hdb_train[mask_hdb]
) if mask_hdb.sum() > 50 else np.nan
print(f"  Silhouette HDBSCAN (sin outliers): {sil_hdb:.3f}  (mayor es mejor)")
print(f"  Davies-Bouldin HDBSCAN           : {db_hdb:.3f}  (menor es mejor)")

ops_train["cluster_hdbscan"] = labels_hdb_train
ops_test["cluster_hdbscan"]  = labels_hdb_test

# Visualización HDBSCAN
fig, ax = plt.subplots(figsize=(8, 6))
cmap_hdb = plt.colormaps["tab10"].resampled(n_clusters_hdb + 1)
colores = [cmap_hdb(l) if l != -1 else (0.5, 0.5, 0.5, 0.3) for l in labels_hdb_train]
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colores, alpha=0.5, s=20)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"HDBSCAN — {n_clusters_hdb} clusters + outliers (gris)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_03_hdbscan_pca.png", dpi=120)
plt.close()
print("Guardado: clustering_03_hdbscan_pca.png")

# --- celda 7 ---
# Perfil de clusters: caracterización por centroide
print("\n=== PERFIL DE CLUSTERS (K-Means) ===")

perfil_clusters = []
for k in range(K_OPTIMO):
    mask = ops_train["cluster_kmeans"] == k
    sub  = ops_train[mask]
    if len(sub) == 0:
        continue
    row = {
        "cluster_kmeans": k,
        "n_operaciones": len(sub),
        "costo_total_median_kpen": round(sub["importe_total_pen"].median() / 1000, 1),
        "n_facturas_median": round(sub["n_facturas"].median(), 1),
        "n_conceptos_median": round(sub["n_conceptos"].median(), 1),
    }
    if "mode" in sub.columns:
        row["modo_dominante"] = sub["mode"].value_counts().index[0] if len(sub) > 0 else "-"
    if "ruta_origen" in sub.columns:
        row["ruta_dominante"] = sub["ruta_origen"].value_counts().index[0] if len(sub) > 0 else "-"
    if "total_contenedores" in sub.columns:
        row["contenedores_median"] = round(sub["total_contenedores"].median(), 1)
    perfil_clusters.append(row)

df_perfil = pd.DataFrame(perfil_clusters)
print(df_perfil.to_string(index=False))
df_perfil.to_csv(REPORTS_DIR / "perfil_clusters.csv", index=False)
print("\nGuardado: reports/perfil_clusters.csv")

# --- celda 8 ---
# PSI (Population Stability Index): detectar drift entre train (2020-2024) y test (2025-2026)
print("\n=== PSI — DETECCIÓN DE DRIFT ===")

TRAIN_PERIOD = "2020-2024 (train)"
TEST_PERIOD  = "2025-2026 (test)"


def calcular_psi_numerico(x_base, x_prod, n_bins=10):
    """PSI para una feature continua usando deciles del conjunto base."""
    x_base = pd.Series(x_base).dropna()
    x_prod = pd.Series(x_prod).dropna()
    if len(x_base) < 10 or len(x_prod) < 5:
        return np.nan
    bins = np.percentile(x_base, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    bins = np.unique(bins)
    if len(bins) < 3:
        return np.nan
    p_base = np.histogram(x_base, bins=bins)[0] / len(x_base)
    p_prod = np.histogram(x_prod, bins=bins)[0] / len(x_prod)
    p_base = np.clip(p_base, 1e-6, None)
    p_prod = np.clip(p_prod, 1e-6, None)
    psi = np.sum((p_prod - p_base) * np.log(p_prod / p_base))
    return float(psi)


def calcular_psi_categorico(x_base, x_prod):
    """PSI para una feature categórica."""
    x_base = pd.Series(x_base).dropna().astype(str)
    x_prod = pd.Series(x_prod).dropna().astype(str)
    cats = set(x_base) | set(x_prod)
    p_base = {c: (x_base == c).mean() for c in cats}
    p_prod = {c: (x_prod == c).mean() for c in cats}
    psi = 0.0
    for c in cats:
        pb = max(p_base.get(c, 0), 1e-6)
        pp = max(p_prod.get(c, 0), 1e-6)
        psi += (pp - pb) * np.log(pp / pb)
    return float(psi)


def interpretar_psi(psi):
    if pd.isna(psi):
        return "SIN DATOS"
    elif psi < 0.10:
        return "ESTABLE"
    elif psi < 0.25:
        return "CAMBIO MODERADO"
    else:
        return "DRIFT CRITICO"


# Features numéricas para PSI
num_psi_cols = [c for c in ["importe_total_pen", "n_facturas", "n_conceptos",
                              "peso_total_kg", "dias_transito_prom"] if c in ops_train.columns]
cat_psi_cols = [c for c in ["mode", "incoterm_grupo", "ruta_origen"] if c in ops_train.columns]

psi_resultados = []

for col in num_psi_cols:
    psi_val = calcular_psi_numerico(ops_train[col].dropna(), ops_test[col].dropna())
    psi_resultados.append({
        "feature": col, "tipo": "numerica",
        "PSI": round(psi_val, 4) if not pd.isna(psi_val) else np.nan,
        "estado": interpretar_psi(psi_val),
    })

for col in cat_psi_cols:
    psi_val = calcular_psi_categorico(ops_train[col], ops_test[col])
    psi_resultados.append({
        "feature": col, "tipo": "categorica",
        "PSI": round(psi_val, 4),
        "estado": interpretar_psi(psi_val),
    })

df_psi = pd.DataFrame(psi_resultados).sort_values("PSI", ascending=False)
print(f"\nResultados PSI ({TRAIN_PERIOD} → {TEST_PERIOD}):")
print(df_psi.to_string(index=False))
df_psi.to_csv(REPORTS_DIR / "psi_baseline_2024_vs_2025.csv", index=False)
print("\nGuardado: reports/psi_baseline_2024_vs_2025.csv")

# Visualización semáforo
fig, ax = plt.subplots(figsize=(8, 5))
colors_map = {"ESTABLE": "green", "CAMBIO MODERADO": "orange", "DRIFT CRITICO": "red", "SIN DATOS": "gray"}
colores_bar = [colors_map.get(e, "gray") for e in df_psi["estado"]]
ax.barh(df_psi["feature"], df_psi["PSI"].fillna(0), color=colores_bar, edgecolor="white")
ax.axvline(0.10, color="orange", linestyle="--", linewidth=1, label="Moderado (0.10)")
ax.axvline(0.25, color="red",    linestyle="--", linewidth=1, label="Crítico (0.25)")
ax.set_xlabel("PSI")
ax.set_title(f"PSI — {TRAIN_PERIOD} vs {TEST_PERIOD}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clustering_04_psi_drift.png", dpi=120)
plt.close()
print("Guardado: clustering_04_psi_drift.png")

# --- celda 9 ---
# Guardar todos los artefactos
all_ops = pd.concat([
    ops_train.assign(split="train"),
    ops_test.assign(split="test"),
], ignore_index=True)

all_ops.to_parquet(PROCESSED_DIR / "operaciones_con_clusters.parquet", index=False)

psi_baseline = {
    "resultados": df_psi,
    "train_period": TRAIN_PERIOD,
    "test_period":  TEST_PERIOD,
    "fecha_calculo": pd.Timestamp.now().isoformat(),
}

joblib.dump(kmeans,         MODELS_DIR / "kmeans_k6.joblib")
joblib.dump(hdbscan_model,  MODELS_DIR / "hdbscan_model.joblib")
joblib.dump(preprocessor,   MODELS_DIR / "clustering_preprocessor.joblib")
joblib.dump(pca,            MODELS_DIR / "clustering_pca15.joblib")
joblib.dump(psi_baseline,   MODELS_DIR / "psi_baseline.joblib")

print(f"\nModelos y datos guardados en {MODELS_DIR}:")
for f in sorted(MODELS_DIR.glob("*.joblib")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
print(f"\nDataset con clusters: {PROCESSED_DIR / 'operaciones_con_clusters.parquet'}")
