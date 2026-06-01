"""
02_eda.py — CRISP-DM Fase 2: Análisis Exploratorio de Datos.

Análisis de calidad, distribuciones, outliers, evolución temporal y correlaciones.
Imputa fechas faltantes. Exporta dataset enriquecido y figuras en alta resolución.

Prerequisito: python src/01_carga_datos.py
Ejecutar    : python src/02_eda.py
Salidas     : data/processed/dataset_eda.parquet
              reports/outliers_resumen.csv
              reports/figures/eda_01_target.png
              reports/figures/eda_02_temporal.png
              reports/figures/eda_03_proveedores.png
              reports/figures/eda_04_modalidad.png
              reports/figures/eda_05_correlaciones.png
              reports/figures/eda_06_calidad_datos.png
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, DPI_FIGURAS, COLOR_PRIMARIO, PALETTE

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
df = pd.read_parquet(PROCESSED_DIR / "dataset_unificado.parquet")
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
campañas = sorted(df["campaña"].dropna().unique().astype(int).tolist())
print(f"Campañas presentes: {campañas}")

# ---------------------------------------------------------------------------
# 1. Calidad de datos
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("1. CALIDAD DE DATOS")
print("=" * 65)

nulos = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
nulos_sig = nulos[nulos > 0]
print(f"\nColumnas con nulos ({len(nulos_sig)} de {df.shape[1]}):")
print(nulos_sig.to_string())

basura = df.astype(str).apply(
    lambda col: col.str.contains(r"#N/D|#REF!|#VALUE!", na=False)
).any(axis=1)
print(f"\nFilas con errores heredados de Excel (#N/D, #REF!): {basura.sum():,}")

dup = df.duplicated(subset=["nro_operacion", "concepto", "importe_total_pen", "fecha_emision"])
print(f"Filas duplicadas exactas: {dup.sum():,}")

cols_clave = ["campaña", "importe_total_pen", "concepto_canonico", "mode",
              "incoterm", "pol", "contenedores", "bultos", "peso_bruto"]
print("\nTipos y nulos de columnas clave:")
print(f"  {'Columna':<28} {'Tipo':<12} {'Nulos':>8} {'%Nulo':>8}")
print("  " + "-" * 58)
for c in cols_clave:
    if c in df.columns:
        n_nul = df[c].isna().sum()
        pct   = n_nul / len(df) * 100
        print(f"  {c:<28} {str(df[c].dtype):<12} {n_nul:>8,} {pct:>7.1f}%")

# Gráfico de calidad: porcentaje de nulos por columna
fig, ax = plt.subplots(figsize=(10, 5))
nulos_sig.head(20).plot.barh(ax=ax, color=COLOR_PRIMARIO, edgecolor="white")
ax.set_xlabel("% de valores nulos")
ax.set_title("Porcentaje de nulos por columna (top 20)", fontsize=13, fontweight="bold")
ax.axvline(10, color="orange", linestyle="--", linewidth=1, label="10%")
ax.axvline(50, color="red",    linestyle="--", linewidth=1, label="50%")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_06_calidad_datos.png", dpi=DPI_FIGURAS)
plt.close()
print("\nGuardado: eda_06_calidad_datos.png")

# ---------------------------------------------------------------------------
# 2. Distribución del target: Importe_Total_PEN
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. DISTRIBUCIÓN DEL TARGET (Importe_Total_PEN)")
print("=" * 65)

target     = df["importe_total_pen"].dropna()
target_pos = target[target > 0]

print(f"  Total filas con target   : {len(target):,}")
print(f"  Positivos                : {len(target_pos):,}")
print(f"  Negativos/cero           : {(target <= 0).sum():,}")
print(f"  Mediana (PEN)            : S/ {target_pos.median():,.0f}")
print(f"  Media   (PEN)            : S/ {target_pos.mean():,.0f}")
print(f"  P10                      : S/ {target_pos.quantile(0.10):,.0f}")
print(f"  P90                      : S/ {target_pos.quantile(0.90):,.0f}")
print(f"  Máximo                   : S/ {target_pos.max():,.0f}")

stats_concepto = (
    df[df["importe_total_pen"] > 0]
    .groupby("concepto_canonico")["importe_total_pen"]
    .agg(["count", "median", "mean", "std"])
    .rename(columns={"count": "n", "median": "mediana", "mean": "media", "std": "desv_std"})
    .sort_values("n", ascending=False)
)
print("\nEstadísticos por concepto canónico:")
print(stats_concepto.round(0).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(np.log1p(target_pos), bins=50, color=COLOR_PRIMARIO, edgecolor="white")
axes[0].set_xlabel("log1p(Importe_Total_PEN)", fontsize=11)
axes[0].set_ylabel("Frecuencia", fontsize=11)
axes[0].set_title("Distribución del target (escala log)", fontsize=12, fontweight="bold")
med_log = np.log1p(target_pos.median())
axes[0].axvline(med_log, color="red", linestyle="--", label=f"Mediana={target_pos.median():,.0f} PEN")
axes[0].legend(fontsize=9)

top_conceptos = stats_concepto.head(10).index.tolist()
data_box = [
    df.loc[(df["concepto_canonico"] == c) & (df["importe_total_pen"] > 0), "importe_total_pen"].values
    for c in top_conceptos
]
axes[1].boxplot(data_box, labels=[c.replace("_", "\n") for c in top_conceptos],
                vert=False, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=COLOR_PRIMARIO, alpha=0.5))
axes[1].set_xscale("log")
axes[1].set_xlabel("Importe_Total_PEN (escala log, PEN)", fontsize=11)
axes[1].set_title("Distribución por concepto (P5–P95)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_01_target.png", dpi=DPI_FIGURAS)
plt.close()
print("\nGuardado: eda_01_target.png")

# ---------------------------------------------------------------------------
# 3. Detección de outliers por concepto — método IQR × 3
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. DETECCIÓN DE OUTLIERS (IQR × 3)")
print("=" * 65)

registros_outlier = []
df["es_outlier"] = False

for concepto, grupo in df.groupby("concepto_canonico"):
    vals_pos = grupo["importe_total_pen"].dropna()
    vals_pos = vals_pos[vals_pos > 0]
    if len(vals_pos) < 10:
        continue
    q1, q3 = vals_pos.quantile([0.25, 0.75])
    iqr        = q3 - q1
    lim_inf    = q1 - 3 * iqr
    lim_sup    = q3 + 3 * iqr
    mask_out   = grupo.index[
        (grupo["importe_total_pen"] < lim_inf) | (grupo["importe_total_pen"] > lim_sup)
    ]
    df.loc[mask_out, "es_outlier"] = True
    if len(mask_out) > 0:
        registros_outlier.append({
            "concepto_canonico": concepto,
            "n_total":           len(vals_pos),
            "n_outliers":        len(mask_out),
            "pct_outliers":      round(len(mask_out) / len(vals_pos) * 100, 1),
            "limite_inferior":   round(lim_inf, 0),
            "limite_superior":   round(lim_sup, 0),
        })

df_outliers = pd.DataFrame(registros_outlier).sort_values("pct_outliers", ascending=False)
print(df_outliers.to_string(index=False))
df_outliers.to_csv(REPORTS_DIR / "outliers_resumen.csv", index=False)
print(f"\nTotal outliers: {df['es_outlier'].sum():,} ({df['es_outlier'].mean()*100:.1f}%)")
print("Guardado: reports/outliers_resumen.csv")

# ---------------------------------------------------------------------------
# 4. Evolución temporal: costo mediano por concepto y campaña
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("4. EVOLUCIÓN TEMPORAL")
print("=" * 65)

df_temporal = (
    df[(df["importe_total_pen"] > 0) & (~df["es_outlier"])]
    .groupby(["campaña", "concepto_canonico"])["importe_total_pen"]
    .median().reset_index()
)
top6 = (
    df.groupby("concepto_canonico")["importe_total_pen"].count()
    .nlargest(6).index.tolist()
)

fig, ax = plt.subplots(figsize=(13, 5))
colors_line = sns.color_palette(PALETTE, n_colors=6)
for i, concepto in enumerate(top6):
    sub = df_temporal[df_temporal["concepto_canonico"] == concepto]
    ax.plot(sub["campaña"], sub["importe_total_pen"], marker="o", linewidth=2,
            label=concepto, color=colors_line[i])
ax.set_xlabel("Campaña", fontsize=11)
ax.set_ylabel("Costo mediano (PEN)", fontsize=11)
ax.set_title("Evolución del costo mediano por concepto (top 6)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S/{x:,.0f}"))
ax.legend(loc="upper left", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_02_temporal.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: eda_02_temporal.png")

# ---------------------------------------------------------------------------
# 5. Top 20 proveedores por costo total
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. TOP 20 PROVEEDORES")
print("=" * 65)

col_prov = "proveedor_principal" if "proveedor_principal" in df.columns else "proveedor"
df_prov  = (
    df[df["importe_total_pen"] > 0]
    .groupby(col_prov)["importe_total_pen"]
    .agg(costo_total="sum", n_facturas="count", costo_mediano="median")
    .sort_values("costo_total", ascending=False).head(20)
)
print(df_prov.round(0).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
df_prov["costo_total"].div(1e6).plot.barh(ax=ax, color=COLOR_PRIMARIO, edgecolor="white")
ax.set_xlabel("Costo total (millones PEN)", fontsize=11)
ax.set_title("Top 20 proveedores por costo total histórico", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for bar in ax.patches:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}M", va="center", fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_03_proveedores.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: eda_03_proveedores.png")

# ---------------------------------------------------------------------------
# 6. Distribución por modalidad de transporte
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("6. DISTRIBUCIÓN POR MODALIDAD DE TRANSPORTE")
print("=" * 65)

if "mode" in df.columns and "type" in df.columns:
    df["modalidad"] = (
        df["mode"].astype(str).str.strip().str.upper()
        + " / "
        + df["type"].astype(str).str.strip().str.upper()
    ).replace({"NAN / NAN": "DESCONOCIDO", "NAN / ": "DESCONOCIDO"})
else:
    df["modalidad"] = "DESCONOCIDO"

df_modal = (
    df[df["importe_total_pen"] > 0]
    .groupby("modalidad")["importe_total_pen"]
    .agg(n="count", mediana="median", total="sum")
    .sort_values("n", ascending=False).head(10)
)
print(df_modal.round(0).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_bar = sns.color_palette(PALETTE, n_colors=len(df_modal))

df_modal["n"].plot.bar(ax=axes[0], color=colors_bar, edgecolor="white")
axes[0].set_title("Número de facturas por modalidad (top 10)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=45)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

df_modal["mediana"].div(1_000).plot.bar(ax=axes[1], color=colors_bar, edgecolor="white")
axes[1].set_title("Costo mediano por modalidad (miles PEN)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=45)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S/{x*1000:,.0f}"))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_04_modalidad.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: eda_04_modalidad.png")

# ---------------------------------------------------------------------------
# 7. Correlaciones entre variables numéricas
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. CORRELACIONES (variables numéricas)")
print("=" * 65)

cols_num = [c for c in ["importe_total_pen", "contenedores", "bultos",
                         "peso_bruto", "monto_total_usd", "igv"] if c in df.columns]
corr_df  = df[cols_num].corr(numeric_only=True)
print(corr_df.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Correlaciones — variables numéricas", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_05_correlaciones.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: eda_05_correlaciones.png")

# ---------------------------------------------------------------------------
# 8. Imputación de fechas y guardado
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("8. IMPUTACIÓN DE FECHAS Y GUARDADO")
print("=" * 65)

df["fecha_imputada"] = df["fecha_emision"].isna()
df["fecha_emision_final"] = df["fecha_emision"].copy()

mask_nula = df["fecha_imputada"]
df.loc[mask_nula, "fecha_emision_final"] = pd.to_datetime(
    df.loc[mask_nula, "campaña"].fillna(2024).astype(int).astype(str) + "-07-01",
    errors="coerce",
)
n_imputadas = mask_nula.sum()
print(f"Fechas imputadas: {n_imputadas:,} ({n_imputadas / len(df) * 100:.1f}%)")
print("Criterio: si fecha_emision es nula → 1 de julio del año de campaña")

out_path = PROCESSED_DIR / "dataset_eda.parquet"
df.to_parquet(out_path, index=False)
print(f"\nGuardado: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
print(f"Figuras exportadas en: {FIGURES_DIR}")
print("\nResumen EDA finalizado. Próximo paso: python src/03_feature_engineering.py")
