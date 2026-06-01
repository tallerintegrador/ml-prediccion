"""
03_feature_engineering.py — CRISP-DM Fase 3: Preparación de datos y construcción de features.

Construye 35+ features con garantía de ausencia de data leakage:
  - Temporales con encoding cíclico
  - Logísticas (días tránsito, densidad, escala de carga)
  - Históricas mediante EWM con halflife en días reales (no en observaciones)
  - Lag genuinos calculados con shift(1) — sin filtración del valor actual
  - Target encoding bayesiano ajustado SOLO en train
  - Features de interacción concepto × modalidad y diferenciación por proveedor
Split temporal respetando cronología: train < 2025-01-01, test ≥ 2025-01-01.

Prerequisito: python src/02_eda.py
Ejecutar    : python src/03_feature_engineering.py
Salidas     : data/processed/dataset_modelable_train.parquet
              data/processed/dataset_modelable_test.parquet
              models/feature_config.joblib
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, MODELS_DIR, FECHA_SPLIT, HALFLIFE_DIAS, RANDOM_STATE
from utils import limpiar_texto, mapear_incoterm_grupo, mapear_ruta_origen, estimar_dias_transito

pd.set_option("display.max_columns", 60)

# ---------------------------------------------------------------------------
# Carga y limpieza inicial
# ---------------------------------------------------------------------------
df = pd.read_parquet(PROCESSED_DIR / "dataset_eda.parquet")
print(f"Dataset EDA cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

n_antes = len(df)
df = df[
    (df["importe_total_pen"] > 0) &
    (df["campaña"].fillna(0) != 2019)
].copy()
print(f"Tras filtros (positivos, sin 2019): {len(df):,} filas ({n_antes - len(df):,} removidas)")

df["fecha_trabajo"] = df["fecha_emision_final"].fillna(
    pd.to_datetime(df["campaña"].astype(int).astype(str) + "-07-01", errors="coerce")
)
df = df.sort_values("fecha_trabajo").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 1. Features temporales
# ---------------------------------------------------------------------------
print("\n[1/9] Construyendo features temporales...")

FECHA_INICIO = pd.Timestamp("2020-01-01")

df["año"]               = df["fecha_trabajo"].dt.year
df["mes"]               = df["fecha_trabajo"].dt.month
df["trimestre"]         = df["fecha_trabajo"].dt.quarter
df["semana_año"]        = df["fecha_trabajo"].dt.isocalendar().week.astype(int)
df["dia_semana"]        = df["fecha_trabajo"].dt.dayofweek
df["dias_desde_inicio"] = (df["fecha_trabajo"] - FECHA_INICIO).dt.days.clip(lower=0)

df["mes_sin"]    = np.sin(2 * np.pi * df["mes"] / 12)
df["mes_cos"]    = np.cos(2 * np.pi * df["mes"] / 12)
df["semana_sin"] = np.sin(2 * np.pi * df["semana_año"] / 52)
df["semana_cos"] = np.cos(2 * np.pi * df["semana_año"] / 52)

df["es_temporada_alta"] = df["mes"].between(8, 12).astype(int)
df["es_cierre_fiscal"]  = df["mes"].isin([11, 12]).astype(int)

mask_imputada = df["fecha_imputada"].fillna(False)
df.loc[mask_imputada, ["mes_sin", "mes_cos", "semana_sin", "semana_cos"]] = 0.0
df.loc[mask_imputada, ["es_temporada_alta", "es_cierre_fiscal"]] = 0

print(f"  OK — fechas imputadas neutralizadas: {mask_imputada.sum():,}")

# ---------------------------------------------------------------------------
# 2. Normalización de entidades categóricas
# ---------------------------------------------------------------------------
print("[2/9] Normalizando entidades categóricas...")

for col in ["proveedor_principal", "proveedor", "agencia_aduana", "acreedor"]:
    if col in df.columns:
        df[f"{col}_norm"] = df[col].apply(limpiar_texto)

# ---------------------------------------------------------------------------
# 3. Features logísticas y de dominio
# ---------------------------------------------------------------------------
print("[3/9] Construyendo features logísticas...")

df["incoterm_grupo"] = df["incoterm"].apply(mapear_incoterm_grupo)
df["ruta_origen"]    = df["pol"].apply(mapear_ruta_origen) if "pol" in df.columns else "OTROS"
df["dias_transito"]  = df["mode"].apply(estimar_dias_transito) if "mode" in df.columns else 25

if "fecha_ata" in df.columns and "fecha_atd" in df.columns:
    dias_reales = (df["fecha_ata"] - df["fecha_atd"]).dt.days
    mask_valido = dias_reales.between(1, 120)
    df.loc[mask_valido, "dias_transito"] = dias_reales[mask_valido]

if "bultos" in df.columns and "contenedores" in df.columns:
    ctnr_pos = df["contenedores"].clip(lower=1).fillna(1)
    df["densidad_bultos"] = (df["bultos"].fillna(0) / ctnr_pos).clip(
        upper=(df["bultos"].fillna(0) / ctnr_pos).quantile(0.99)
    )
else:
    df["densidad_bultos"] = 0.0
    ctnr_pos = pd.Series(1, index=df.index)

if "peso_bruto" in df.columns and "contenedores" in df.columns:
    df["carga_peso"] = (df["peso_bruto"].fillna(0) / ctnr_pos).clip(
        upper=(df["peso_bruto"].fillna(0) / ctnr_pos).quantile(0.99)
    )
else:
    df["carga_peso"] = 0.0

col_proyect = "proyect" if "proyect" in df.columns else None
if col_proyect:
    df["tiene_proyecto"] = (
        df[col_proyect].notna() &
        (df[col_proyect].astype(str).str.strip() != "") &
        (df[col_proyect].astype(str).str.lower() != "nan")
    ).astype(int)
else:
    df["tiene_proyecto"] = 0

for raw_col, log_col in [("contenedores", "log_contenedores"),
                          ("bultos",        "log_bultos"),
                          ("peso_bruto",    "log_peso_bruto")]:
    if raw_col in df.columns:
        df[log_col] = np.log1p(df[raw_col].clip(lower=0).fillna(0))
    else:
        df[log_col] = 0.0

print(f"  OK — incoterm_grupo, ruta_origen, densidad_bultos, carga_peso, log_*")

# ---------------------------------------------------------------------------
# 4. Target log y máscara de train
# ---------------------------------------------------------------------------
FECHA_SPLIT_TS = pd.Timestamp(FECHA_SPLIT)
train_mask     = df["fecha_trabajo"] < FECHA_SPLIT_TS

df["log_target"] = np.log1p(df["importe_total_pen"])
mediana_global_log = float(df.loc[train_mask, "log_target"].median())

col_prov = "proveedor_norm" if "proveedor_norm" in df.columns else "proveedor"

print(f"\n  Máscara train: {train_mask.sum():,} | test: {(~train_mask).sum():,}")

# ---------------------------------------------------------------------------
# 5. Tarifa histórica EWM con halflife en días reales (anti-leakage: shift 1)
# ---------------------------------------------------------------------------
print(f"\n[4/9] Calculando tarifa histórica EWM (halflife={HALFLIFE_DIAS}D reales, shift=1)...")

def ewm_shift1(grupo, col, halflife_d):
    """EWM sobre log_target con shift(1) para evitar leakage."""
    return (
        grupo[col]
        .ewm(halflife=f"{halflife_d}D", times=pd.DatetimeIndex(grupo["fecha_trabajo"]), min_periods=1)
        .mean()
        .shift(1)
    )

ewm_prov_conc = (
    df.groupby([col_prov, "concepto_canonico"], group_keys=False)
    .apply(lambda g: ewm_shift1(g, "log_target", HALFLIFE_DIAS))
)
ewm_conc = (
    df.groupby("concepto_canonico", group_keys=False)
    .apply(lambda g: ewm_shift1(g, "log_target", HALFLIFE_DIAS))
)
ewm_global = (
    df["log_target"]
    .ewm(halflife=f"{HALFLIFE_DIAS}D", times=pd.DatetimeIndex(df["fecha_trabajo"]), min_periods=1)
    .mean().shift(1)
)

df["tarifa_historica"] = (
    ewm_prov_conc
    .where(ewm_prov_conc.notna(), ewm_conc)
    .where(ewm_conc.notna(), ewm_global)
    .fillna(mediana_global_log)
)
print(f"  tarifa_historica corr con log_target: {df['tarifa_historica'].corr(df['log_target']):.3f}")

# Tarifa normalizada por contenedor
if "contenedores" in df.columns:
    df["_log_costo_ctnr"] = np.log1p(
        df["importe_total_pen"] / df["contenedores"].clip(lower=1).fillna(1)
    )
    ewm_ctnr_prov = (
        df.groupby([col_prov, "concepto_canonico"], group_keys=False)
        .apply(lambda g: ewm_shift1(g, "_log_costo_ctnr", HALFLIFE_DIAS))
    )
    ewm_ctnr_conc = (
        df.groupby("concepto_canonico", group_keys=False)
        .apply(lambda g: ewm_shift1(g, "_log_costo_ctnr", HALFLIFE_DIAS))
    )
    df["tarifa_por_contenedor"] = (
        ewm_ctnr_prov.where(ewm_ctnr_prov.notna(), ewm_ctnr_conc)
        .fillna(df["tarifa_historica"])
    )
    df.drop(columns=["_log_costo_ctnr"], inplace=True)
else:
    df["tarifa_por_contenedor"] = df["tarifa_historica"]

# ---------------------------------------------------------------------------
# 6. Lag features — sin leakage (shift estricto)
# ---------------------------------------------------------------------------
print("[5/9] Calculando lag features con shift(1/2/3)...")

def lag_prov_conc(df_, col, n, fallback_col):
    prov_conc = (
        df_.groupby([col_prov, "concepto_canonico"], group_keys=False)[col]
        .transform(lambda x: x.shift(n))
    )
    concepto_lag = (
        df_.groupby("concepto_canonico", group_keys=False)[col]
        .transform(lambda x: x.shift(n))
    )
    return prov_conc.where(prov_conc.notna(), concepto_lag).fillna(mediana_global_log)

df["lag1_costo"]      = lag_prov_conc(df, "log_target", 1, "log_target")
df["lag2_costo"]      = lag_prov_conc(df, "log_target", 2, "log_target")
df["lag3_costo"]      = lag_prov_conc(df, "log_target", 3, "log_target")
df["rolling_median_3"] = (
    df.groupby([col_prov, "concepto_canonico"], group_keys=False)["log_target"]
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).median())
    .fillna(mediana_global_log)
)

print(f"  lag1 corr: {df['lag1_costo'].corr(df['log_target']):.3f}  "
      f"| lag2: {df['lag2_costo'].corr(df['log_target']):.3f}  "
      f"| rolling3: {df['rolling_median_3'].corr(df['log_target']):.3f}")

df["frecuencia_proveedor"] = df.groupby(col_prov).cumcount()

# ---------------------------------------------------------------------------
# 7. Target encoding bayesiano — FIT SOLO EN TRAIN
# ---------------------------------------------------------------------------
print("[6/9] Target encoding bayesiano (fit en train, aplica a todo df)...")

SMOOTHING = 10

def te_bayesiano_fit(df_train, group_series_full, target_col="log_target", smoothing=SMOOTHING):
    """Bayesian smoothed TE: ajusta en train, devuelve dict para aplicar en todo el df."""
    global_mean = float(df_train[target_col].mean())
    g_train     = group_series_full[df_train.index].rename("_gk")
    stats       = df_train.groupby(g_train)[target_col].agg(["mean", "count"])
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    )
    return stats["smoothed"].to_dict(), global_mean

df_train_fe = df[train_mask]

te_ci_key   = df["concepto_canonico"].astype(str) + "_" + df["incoterm_grupo"].astype(str)
te_ci_map, te_ci_global = te_bayesiano_fit(df_train_fe, te_ci_key)
df["te_concepto_incoterm"] = te_ci_key.map(te_ci_map).fillna(te_ci_global)

te_cr_key   = df["concepto_canonico"].astype(str) + "_" + df["ruta_origen"].astype(str)
te_cr_map, te_cr_global = te_bayesiano_fit(df_train_fe, te_cr_key)
df["te_concepto_ruta"] = te_cr_key.map(te_cr_map).fillna(te_cr_global)

te_prov_map, te_prov_global = te_bayesiano_fit(df_train_fe, df[col_prov], smoothing=20)
df["te_proveedor"] = df[col_prov].map(te_prov_map).fillna(te_prov_global)

print(f"  te_concepto_incoterm corr: {df['te_concepto_incoterm'].corr(df['log_target']):.3f}")
print(f"  te_concepto_ruta corr    : {df['te_concepto_ruta'].corr(df['log_target']):.3f}")
print(f"  te_proveedor corr        : {df['te_proveedor'].corr(df['log_target']):.3f}")

# ---------------------------------------------------------------------------
# 8. Features de interacción y diferenciación
# ---------------------------------------------------------------------------
print("[7/9] Calculando features de interacción y diferenciación...")

mediana_por_concepto = df_train_fe.groupby("concepto_canonico")["log_target"].median()

mediana_conc_mode = (
    df_train_fe.groupby(["concepto_canonico", "mode"])["log_target"]
    .median().reset_index()
    .rename(columns={"log_target": "median_concepto_mode"})
)
df = df.merge(mediana_conc_mode, on=["concepto_canonico", "mode"], how="left")
df["median_concepto_mode"] = df["median_concepto_mode"].fillna(mediana_global_log)

df["diff_tarifa_mediana"]  = (
    df["tarifa_historica"] - df["concepto_canonico"].map(mediana_por_concepto)
).fillna(0.0)

df["provider_premium_idx"] = (
    df["lag1_costo"] - df["concepto_canonico"].map(mediana_por_concepto)
).fillna(0.0)

df["n_obs_prov_concepto"] = df.groupby([col_prov, "concepto_canonico"]).cumcount()

concepto_cv = (
    df_train_fe.groupby("concepto_canonico")["log_target"]
    .agg(lambda x: float(x.std() / x.mean()) if len(x) > 1 and x.mean() != 0 else 0.0)
)
df["concepto_cv"] = df["concepto_canonico"].map(concepto_cv).fillna(0.0)

print(f"  provider_premium_idx corr: {df['provider_premium_idx'].corr(df['log_target']):.3f}")
print(f"  n_obs_prov_concepto corr : {df['n_obs_prov_concepto'].corr(df['log_target']):.3f}")
print(f"  concepto_cv corr         : {df['concepto_cv'].corr(df['log_target']):.3f}")

# ---------------------------------------------------------------------------
# 9. Pesos de muestra (outliers y fechas imputadas ponderados, no eliminados)
# ---------------------------------------------------------------------------
print("[8/9] Asignando sample_weight...")

es_outlier = df.get("es_outlier", pd.Series(False, index=df.index)).fillna(False)
df["sample_weight"] = np.where(es_outlier, 0.15, np.where(mask_imputada, 0.3, 1.0))

print(f"  real(1.0): {(df['sample_weight'] == 1.0).sum():,} | "
      f"imputada(0.3): {(df['sample_weight'] == 0.3).sum():,} | "
      f"outlier(0.15): {(df['sample_weight'] == 0.15).sum():,}")

# ---------------------------------------------------------------------------
# Lista definitiva de features
# ---------------------------------------------------------------------------
FEATURES_NUM = [
    "año", "mes", "trimestre", "semana_año", "dia_semana",
    "dias_desde_inicio", "mes_sin", "mes_cos", "semana_sin", "semana_cos",
    "es_temporada_alta", "es_cierre_fiscal",
    "dias_transito", "densidad_bultos", "carga_peso", "tiene_proyecto",
    "tarifa_historica",
    "lag1_costo", "lag2_costo", "lag3_costo", "rolling_median_3",
    "frecuencia_proveedor",
    "te_concepto_incoterm", "te_concepto_ruta", "te_proveedor",
    "median_concepto_mode", "diff_tarifa_mediana",
    "provider_premium_idx", "n_obs_prov_concepto", "concepto_cv",
    "log_contenedores", "log_bultos", "log_peso_bruto",
    "tarifa_por_contenedor",
]

FEATURES_CAT = [
    col for col in [
        "concepto_canonico", "incoterm_grupo", "ruta_origen",
        "proveedor_norm" if "proveedor_norm" in df.columns else "proveedor",
        "agencia_aduana_norm" if "agencia_aduana_norm" in df.columns else "agencia_aduana",
        "mode", "type",
    ] if col in df.columns
]

FEATURES_NUM = [f for f in FEATURES_NUM if f in df.columns]
FEATURES_CAT = [f for f in FEATURES_CAT if f in df.columns]

faltantes = [f for f in FEATURES_NUM + FEATURES_CAT if f not in df.columns]
if faltantes:
    print(f"ADVERTENCIA — features no encontradas: {faltantes}")

print(f"\n  Features numéricas ({len(FEATURES_NUM)}): {FEATURES_NUM[:10]} ...")
print(f"  Features categóricas ({len(FEATURES_CAT)}): {FEATURES_CAT}")

# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------
print(f"\n[9/9] Split temporal en {FECHA_SPLIT}...")

train = df[df["fecha_trabajo"] < FECHA_SPLIT_TS].copy()
test  = df[df["fecha_trabajo"] >= FECHA_SPLIT_TS].copy()

print(f"  Train: {len(train):,} filas  ({train['fecha_trabajo'].min().date()} → {train['fecha_trabajo'].max().date()})")
print(f"  Test : {len(test):,} filas   ({test['fecha_trabajo'].min().date()} → {test['fecha_trabajo'].max().date()})")

for nombre, subdf in [("TRAIN", train), ("TEST", test)]:
    t = subdf["importe_total_pen"]
    print(f"  {nombre} target — mediana: S/{t.median():,.0f} | media: S/{t.mean():,.0f} | n={len(t):,}")

# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------
feature_config = {
    "features_num":          FEATURES_NUM,
    "features_cat":          FEATURES_CAT,
    "target":                "importe_total_pen",
    "col_prov":              col_prov,
    "mediana_global_log":    mediana_global_log,
    "te_ci_map":             te_ci_map,
    "te_ci_global":          te_ci_global,
    "te_cr_map":             te_cr_map,
    "te_cr_global":          te_cr_global,
    "te_prov_map":           te_prov_map,
    "te_prov_global":        te_prov_global,
    "mediana_por_concepto":  mediana_por_concepto.to_dict(),
    "concepto_cv":           concepto_cv.to_dict(),
    "fecha_split":           FECHA_SPLIT,
    "halflife_dias":         HALFLIFE_DIAS,
}

train.to_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet", index=False)
test.to_parquet(PROCESSED_DIR  / "dataset_modelable_test.parquet",  index=False)
joblib.dump(feature_config, MODELS_DIR / "feature_config.joblib")

print(f"\n  Guardado: dataset_modelable_train.parquet  ({(PROCESSED_DIR/'dataset_modelable_train.parquet').stat().st_size/1024:.0f} KB)")
print(f"  Guardado: dataset_modelable_test.parquet   ({(PROCESSED_DIR/'dataset_modelable_test.parquet').stat().st_size/1024:.0f} KB)")
print(f"  Guardado: models/feature_config.joblib")
print("\nFeature Engineering finalizado. Próximo paso: python src/04a_regresion.py")
