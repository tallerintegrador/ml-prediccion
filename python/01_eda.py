"""
01_eda.py — Análisis exploratorio del dataset unificado de importaciones.

Prerequisito: python python/setup_y_carga.py
Ejecutar    : python python/01_eda.py
Salidas     : data/processed/dataset_eda.parquet
              reports/outliers_resumen.csv
              reports/figures/eda_*.png (5 gráficos)
"""

# --- celda 1 ---
# Imports
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sin interfaz gráfica para ejecución en terminal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

# --- celda 2 ---
# Rutas
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Carga del dataset unificado
df = pd.read_parquet(PROCESSED_DIR / "dataset_unificado.parquet")
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"Campañas presentes: {sorted(df['campaña'].dropna().unique().astype(int).tolist())}")

# --- celda 3 ---
# Calidad de datos: nulos y valores problemáticos heredados del Excel
print("\n=== CALIDAD DE DATOS ===")

# % nulos por columna (solo columnas con >0% nulos)
nulos = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
nulos_significativos = nulos[nulos > 0]
print(f"\nColumnas con nulos ({len(nulos_significativos)} de {df.shape[1]}):")
print(nulos_significativos.to_string())

# Valores basura heredados del Excel
basura_mask = df.astype(str).apply(
    lambda col: col.str.contains(r"#N/D|#REF!|#VALUE!", na=False)
).any(axis=1)
print(f"\nFilas con valores '#N/D'/'#REF!' heredados: {basura_mask.sum():,}")

# Filas duplicadas
dup = df.duplicated(subset=["nro_operacion", "concepto", "importe_total_pen", "fecha_emision"])
print(f"Filas duplicadas (misma operación + concepto + importe + fecha): {dup.sum():,}")

# Tipos de dato de columnas clave
print("\nTipos de dato (columnas clave):")
cols_clave = ["campaña", "importe_total_pen", "concepto_canonico", "mode", "incoterm",
              "pol", "contenedores", "bultos", "peso_bruto"]
for c in cols_clave:
    if c in df.columns:
        print(f"  {c:<25} {str(df[c].dtype):<12}  nulos={df[c].isna().sum()}")

# --- celda 4 ---
# Distribución del target: Importe_Total_PEN
print("\n=== DISTRIBUCIÓN DEL TARGET (Importe_Total_PEN) ===")

target = df["importe_total_pen"].dropna()
target_pos = target[target > 0]

print(f"  Total filas con target: {len(target):,}")
print(f"  Positivos            : {len(target_pos):,}")
print(f"  Negativos/cero       : {(target <= 0).sum():,}")
print(f"  Mediana (PEN)        : S/ {target_pos.median():,.0f}")
print(f"  Media   (PEN)        : S/ {target_pos.mean():,.0f}")
print(f"  P10                  : S/ {target_pos.quantile(0.10):,.0f}")
print(f"  P90                  : S/ {target_pos.quantile(0.90):,.0f}")
print(f"  Máximo               : S/ {target_pos.max():,.0f}")

print("\nEstadísticos por concepto canónico:")
stats_concepto = (
    df[df["importe_total_pen"] > 0]
    .groupby("concepto_canonico")["importe_total_pen"]
    .agg(["count", "median", "mean", "std"])
    .rename(columns={"count": "n", "median": "mediana", "mean": "media", "std": "desv_std"})
    .sort_values("n", ascending=False)
)
print(stats_concepto.round(0).to_string())

# Gráfico 1: distribución del target (histograma log-scale + boxplots por concepto)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(np.log1p(target_pos), bins=50, color="steelblue", edgecolor="white")
axes[0].set_xlabel("log1p(Importe_Total_PEN)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Distribución del target (escala log)")

top_conceptos = stats_concepto.head(8).index.tolist()
data_box = [
    df.loc[(df["concepto_canonico"] == c) & (df["importe_total_pen"] > 0), "importe_total_pen"].values
    for c in top_conceptos
]
axes[1].boxplot(data_box, labels=top_conceptos, vert=False, showfliers=False)
axes[1].set_xscale("log")
axes[1].set_xlabel("Importe_Total_PEN (log)")
axes[1].set_title("Distribución por concepto (sin outliers extremos)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_01_distribucion_target.png", dpi=120)
plt.close()
print("\nGuardado: eda_01_distribucion_target.png")

# --- celda 5 ---
# Detección de outliers por concepto: método IQR × 3
print("\n=== DETECCIÓN DE OUTLIERS (IQR × 3) ===")

registros_outlier = []
df["es_outlier"] = False

for concepto, grupo in df.groupby("concepto_canonico"):
    vals = grupo["importe_total_pen"].dropna()
    vals_pos = vals[vals > 0]
    if len(vals_pos) < 10:
        continue
    q1, q3 = vals_pos.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_inf = q1 - 3 * iqr
    limite_sup = q3 + 3 * iqr
    mask_outlier = (
        grupo.index[
            (grupo["importe_total_pen"] < limite_inf) |
            (grupo["importe_total_pen"] > limite_sup)
        ]
    )
    df.loc[mask_outlier, "es_outlier"] = True
    n_out = len(mask_outlier)
    if n_out > 0:
        registros_outlier.append({
            "concepto_canonico": concepto,
            "n_total": len(vals_pos),
            "n_outliers": n_out,
            "pct_outliers": round(n_out / len(vals_pos) * 100, 1),
            "limite_inferior": round(limite_inf, 0),
            "limite_superior": round(limite_sup, 0),
        })

df_outliers = pd.DataFrame(registros_outlier).sort_values("pct_outliers", ascending=False)
print(df_outliers.to_string(index=False))
df_outliers.to_csv(REPORTS_DIR / "outliers_resumen.csv", index=False)
print(f"\nTotal outliers marcados: {df['es_outlier'].sum():,} ({df['es_outlier'].mean()*100:.1f}%)")

# --- celda 6 ---
# Evolución temporal: costo mediano por concepto y campaña
print("\n=== EVOLUCIÓN TEMPORAL ===")

df_temporal = (
    df[(df["importe_total_pen"] > 0) & (~df["es_outlier"])]
    .groupby(["campaña", "concepto_canonico"])["importe_total_pen"]
    .median()
    .reset_index()
)

top6 = (
    df.groupby("concepto_canonico")["importe_total_pen"]
    .count()
    .nlargest(6)
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(12, 5))
for concepto in top6:
    sub = df_temporal[df_temporal["concepto_canonico"] == concepto]
    ax.plot(sub["campaña"], sub["importe_total_pen"], marker="o", label=concepto)
ax.set_xlabel("Campaña")
ax.set_ylabel("Costo mediano (PEN)")
ax.set_title("Evolución del costo mediano por concepto (top 6)")
ax.legend(loc="upper left", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S/{x:,.0f}"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_02_evolucion_temporal.png", dpi=120)
plt.close()
print("Guardado: eda_02_evolucion_temporal.png")

# --- celda 7 ---
# Análisis de proveedores: top 20 por costo total
print("\n=== TOP 20 PROVEEDORES (PROVEEDOR PRINCIPAL) ===")

col_prov = "proveedor_principal" if "proveedor_principal" in df.columns else "proveedor"

df_proveedores = (
    df[df["importe_total_pen"] > 0]
    .groupby(col_prov)["importe_total_pen"]
    .agg(["sum", "count", "median"])
    .rename(columns={"sum": "costo_total", "count": "n_facturas", "median": "costo_mediano"})
    .sort_values("costo_total", ascending=False)
    .head(20)
)
print(df_proveedores.round(0).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
df_proveedores["costo_total"].div(1e6).plot.barh(ax=ax, color="teal", edgecolor="white")
ax.set_xlabel("Costo total (millones PEN)")
ax.set_title("Top 20 proveedores principales por costo total histórico")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_03_top_proveedores.png", dpi=120)
plt.close()
print("Guardado: eda_03_top_proveedores.png")

# --- celda 8 ---
# Modalidad de transporte: distribución de costos
print("\n=== DISTRIBUCIÓN POR MODALIDAD DE TRANSPORTE ===")

if "mode" in df.columns and "type" in df.columns:
    df["modalidad"] = (
        df["mode"].astype(str).str.strip().str.upper()
        + "_"
        + df["type"].astype(str).str.strip().str.upper()
    )
    df["modalidad"] = df["modalidad"].replace({"NAN_NAN": "DESCONOCIDO", "NAN_": "DESCONOCIDO"})
else:
    df["modalidad"] = "DESCONOCIDO"

df_modal = (
    df[df["importe_total_pen"] > 0]
    .groupby("modalidad")["importe_total_pen"]
    .agg(n="count", mediana="median", total="sum")
    .sort_values("n", ascending=False)
    .head(10)
)
print(df_modal.round(0).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df_modal["n"].plot.bar(ax=axes[0], color="coral", edgecolor="white")
axes[0].set_title("Número de facturas por modalidad (top 10)")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=45)

df_modal["mediana"].plot.bar(ax=axes[1], color="goldenrod", edgecolor="white")
axes[1].set_title("Costo mediano por modalidad (PEN)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S/{x:,.0f}"))
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_04_modalidad_transporte.png", dpi=120)
plt.close()
print("Guardado: eda_04_modalidad_transporte.png")

# --- celda 9 ---
# Correlaciones entre variables numéricas
print("\n=== CORRELACIONES (variables numéricas) ===")

cols_num = [c for c in ["importe_total_pen", "contenedores", "bultos", "peso_bruto",
                         "monto_total_usd", "igv"] if c in df.columns]

corr_df = df[cols_num].corr(numeric_only=True)
print(corr_df.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, ax=ax, square=True
)
ax.set_title("Correlaciones — variables numéricas")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_05_correlaciones.png", dpi=120)
plt.close()
print("Guardado: eda_05_correlaciones.png")

# --- celda 10 ---
# Imputación de fechas faltantes y guardado del parquet de EDA
print("\n=== IMPUTACIÓN DE FECHAS Y GUARDADO ===")

# Si fecha_emision es nula, imputar al 1 de julio del año de campaña
df["fecha_imputada"] = df["fecha_emision"].isna()
df["fecha_emision_final"] = df["fecha_emision"].copy()

mask_nula = df["fecha_imputada"]
df.loc[mask_nula, "fecha_emision_final"] = pd.to_datetime(
    df.loc[mask_nula, "campaña"].astype(int).astype(str) + "-07-01",
    errors="coerce",
)
n_imputadas = mask_nula.sum()
pct_imputadas = n_imputadas / len(df) * 100
print(f"Fechas imputadas: {n_imputadas:,} ({pct_imputadas:.1f}%)")

out_path = PROCESSED_DIR / "dataset_eda.parquet"
df.to_parquet(out_path, index=False)
print(f"Guardado: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
print(f"\nFiguras guardadas en: {FIGURES_DIR}")
