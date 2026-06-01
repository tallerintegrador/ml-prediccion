"""
demo_prediccion.py — Demo de predicción end-to-end de costos de importación.

Prerequisito: ejecutar todos los scripts anteriores en orden.
Ejecutar    : python python/demo_prediccion.py
Salidas     : reports/figures/demo_residuos.png
"""

# --- celda 1 ---
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    limpiar_texto as limpiar,
    mapear_incoterm_grupo,
    mapear_ruta_origen,
    estimar_dias_transito,
    INCOTERM_GRUPO,
    RUTA_ORIGEN,
    DIAS_TRANSITO,
    mediana_porcentaje_error_absoluto,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR      = Path(__file__).parent.parent
MODELS_DIR    = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR   = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

modelos_requeridos = [
    "lgb_multi_concepto.joblib",
    "lgb_p10_multi_concepto.joblib",
    "lgb_p90_multi_concepto.joblib",
    "bias_correction_escalar.joblib",
    "bias_correction_piecewise.joblib",
    "bias_prov_concepto.joblib",
    "cqr_margenes_por_concepto.joblib",
    "regresion_medians.joblib",
    "feature_config.joblib",
    "isolation_forest.joblib",
    "riesgo_encoder.joblib",
    "riesgo_score_params.joblib",
]

faltantes = [m for m in modelos_requeridos if not (MODELS_DIR / m).exists()]
if faltantes:
    print(f"ERROR: modelos faltantes en {MODELS_DIR}:")
    for m in faltantes:
        print(f"  - {m}")
    print("\nEjecuta primero: 02_feature_engineering.py, 03a_regresion.py y 03b_clasificacion.py")
    raise SystemExit(1)

modelos_por_concepto = joblib.load(MODELS_DIR / "lgb_multi_concepto.joblib")
modelos_p10          = joblib.load(MODELS_DIR / "lgb_p10_multi_concepto.joblib")
modelos_p90          = joblib.load(MODELS_DIR / "lgb_p90_multi_concepto.joblib")
bias_escalar         = joblib.load(MODELS_DIR / "bias_correction_escalar.joblib")
bias_piecewise       = joblib.load(MODELS_DIR / "bias_correction_piecewise.joblib")
bias_prov_concepto   = joblib.load(MODELS_DIR / "bias_prov_concepto.joblib")
cqr_margenes         = joblib.load(MODELS_DIR / "cqr_margenes_por_concepto.joblib")
regresion_medians    = joblib.load(MODELS_DIR / "regresion_medians.joblib")
feature_config       = joblib.load(MODELS_DIR / "feature_config.joblib")
iso_forest           = joblib.load(MODELS_DIR / "isolation_forest.joblib")
riesgo_encoder       = joblib.load(MODELS_DIR / "riesgo_encoder.joblib")
riesgo_score_params  = joblib.load(MODELS_DIR / "riesgo_score_params.joblib")

FEATURES_NUM       = feature_config["features_num"]
FEATURES_CAT       = feature_config["features_cat"]
ALL_FEATURES       = FEATURES_NUM + FEATURES_CAT
CONCEPTOS          = sorted(modelos_por_concepto.keys())
COL_PROV           = feature_config.get("col_prov", "proveedor_norm")
MEDIANA_GLOBAL_LOG = feature_config.get("mediana_global_log", 7.0)
global_mean_log    = MEDIANA_GLOBAL_LOG

_TE_CI_MAP    = feature_config.get("te_ci_map", {})
_TE_CI_GLOBAL = feature_config.get("te_ci_global", MEDIANA_GLOBAL_LOG)
_TE_CR_MAP    = feature_config.get("te_cr_map", {})
_TE_CR_GLOBAL = feature_config.get("te_cr_global", MEDIANA_GLOBAL_LOG)
_TE_PROV_MAP  = feature_config.get("te_prov_map", {})
_TE_PROV_GLOBAL = feature_config.get("te_prov_global", MEDIANA_GLOBAL_LOG)
_MEDIANA_POR_CONCEPTO = feature_config.get("mediana_por_concepto", {})
_CONCEPTO_CV          = feature_config.get("concepto_cv", {})

print(f"Modelos cargados: {len(modelos_por_concepto)} conceptos")
print(f"Conceptos disponibles: {CONCEPTOS}")
print(f"TE maps: {len(_TE_CI_MAP)} concepto×incoterm | {len(_TE_PROV_MAP)} proveedores")

# --- celda 2 ---
# Precomputar estadísticos históricos desde el train set
print("\nPrecomputando estadísticos históricos desde train set...")

FECHA_INICIO = pd.Timestamp("2020-01-01")

TARIFAS_PROV_CONCEPTO  = {}
TARIFAS_CONCEPTO       = {}
TARIFA_GLOBAL          = MEDIANA_GLOBAL_LOG
LAG1_PROV_CONCEPTO     = {}
LAG1_CONCEPTO          = {}
LAG2_PROV_CONCEPTO     = {}
LAG3_PROV_CONCEPTO     = {}
ROLLING3_PROV_CONCEPTO = {}
FRECUENCIA_PROVEEDOR   = {}
N_OBS_PROV_CONCEPTO    = {}
TE_PROVEEDOR          = {str(k).upper(): v for k, v in _TE_PROV_MAP.items()}
TE_CONCEPTO_INCOTERM  = _TE_CI_MAP
TE_CONCEPTO_RUTA      = _TE_CR_MAP
MEDIANA_POR_CONCEPTO  = {k: float(v) for k, v in _MEDIANA_POR_CONCEPTO.items()}
CONCEPTO_CV           = {k: float(v) for k, v in _CONCEPTO_CV.items()}
MEDIANA_CONCEPTO_MODE = {}
TARIFA_POR_CONTENEDOR_MAP     = {}
TARIFA_POR_CONTENEDOR_CONCEPTO = {}

if (PROCESSED_DIR / "dataset_modelable_train.parquet").exists():
    train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
    train["log_target"] = np.log1p(train["importe_total_pen"])
    if "fecha_trabajo" in train.columns:
        train = train.sort_values("fecha_trabajo")

    col_prov = COL_PROV if COL_PROV in train.columns else "proveedor"
    global_mean_log = float(train["log_target"].mean())

    for (prov, concepto), grupo in train.groupby([col_prov, "concepto_canonico"]):
        prov_key = (str(prov).upper(), concepto)
        if "fecha_trabajo" in grupo.columns:
            vals = grupo["log_target"].ewm(
                halflife="365D",
                times=pd.DatetimeIndex(grupo["fecha_trabajo"]),
                min_periods=1,
            ).mean()
        else:
            vals = grupo["log_target"].ewm(halflife=365, min_periods=1).mean()
        TARIFAS_PROV_CONCEPTO[prov_key] = float(vals.iloc[-1])
        hist = grupo["log_target"].values
        LAG1_PROV_CONCEPTO[prov_key]     = float(hist[-1])
        LAG2_PROV_CONCEPTO[prov_key]     = float(hist[-2]) if len(hist) >= 2 else float(hist[-1])
        LAG3_PROV_CONCEPTO[prov_key]     = float(hist[-3]) if len(hist) >= 3 else float(hist[-1])
        ROLLING3_PROV_CONCEPTO[prov_key] = float(grupo["log_target"].tail(3).median())
        N_OBS_PROV_CONCEPTO[prov_key]    = int(len(hist))

    for concepto, grupo in train.groupby("concepto_canonico"):
        if "fecha_trabajo" in grupo.columns:
            vals = grupo["log_target"].ewm(
                halflife="365D",
                times=pd.DatetimeIndex(grupo["fecha_trabajo"]),
                min_periods=1,
            ).mean()
        else:
            vals = grupo["log_target"].ewm(halflife=365, min_periods=1).mean()
        TARIFAS_CONCEPTO[concepto] = float(vals.iloc[-1])
        LAG1_CONCEPTO[concepto]    = float(grupo["log_target"].iloc[-1])

    TARIFA_GLOBAL = float(train["log_target"].ewm(halflife=365, min_periods=1).mean().iloc[-1])

    FRECUENCIA_PROVEEDOR = {
        str(k).upper(): int(v)
        for k, v in train.groupby(col_prov).size().items()
    }

    MEDIANA_CONCEPTO_MODE = (
        train.groupby(["concepto_canonico", "mode"])["log_target"].median().to_dict()
        if "mode" in train.columns else {}
    )

    if "contenedores" in train.columns:
        train["_log_costo_ctnr"] = np.log1p(
            train["importe_total_pen"] / train["contenedores"].clip(lower=1).fillna(1)
        )
        for (prov, concepto), grupo in train.groupby([col_prov, "concepto_canonico"]):
            prov_key = (str(prov).upper(), concepto)
            if "fecha_trabajo" in grupo.columns:
                vals = grupo["_log_costo_ctnr"].ewm(
                    halflife="365D",
                    times=pd.DatetimeIndex(grupo["fecha_trabajo"]),
                    min_periods=1,
                ).mean()
            else:
                vals = grupo["_log_costo_ctnr"].ewm(halflife=365, min_periods=1).mean()
            TARIFA_POR_CONTENEDOR_MAP[prov_key] = float(vals.iloc[-1])
        for concepto, grupo in train.groupby("concepto_canonico"):
            if "fecha_trabajo" in grupo.columns:
                vals = grupo["_log_costo_ctnr"].ewm(
                    halflife="365D",
                    times=pd.DatetimeIndex(grupo["fecha_trabajo"]),
                    min_periods=1,
                ).mean()
            else:
                vals = grupo["_log_costo_ctnr"].ewm(halflife=365, min_periods=1).mean()
            TARIFA_POR_CONTENEDOR_CONCEPTO[concepto] = float(vals.iloc[-1])
        train.drop(columns=["_log_costo_ctnr"], inplace=True)

    print(f"  Tarifas prov×concepto : {len(TARIFAS_PROV_CONCEPTO)}")
    print(f"  Tarifas por concepto  : {len(TARIFAS_CONCEPTO)}")
    print(f"  Proveedores en train  : {len(FRECUENCIA_PROVEEDOR)}")
else:
    print("  (train no disponible — usando medianas de fallback)")

# --- celda 3 ---
# Funciones de lookup con fallback seguro para valores 0.0
# Se usa "is not None" explícito para evitar que 0.0 sea tratado como falsy.

def _lookup(primary_dict, primary_key, fallback_dict, fallback_key, default):
    val = primary_dict.get(primary_key)
    if val is not None:
        return val
    val = fallback_dict.get(fallback_key)
    return val if val is not None else default


def obtener_tarifa_historica(proveedor: str, concepto: str) -> float:
    prov = limpiar(proveedor)
    return _lookup(TARIFAS_PROV_CONCEPTO, (prov, concepto),
                   TARIFAS_CONCEPTO, concepto, TARIFA_GLOBAL)


def obtener_lag1(proveedor: str, concepto: str) -> float:
    prov = limpiar(proveedor)
    return _lookup(LAG1_PROV_CONCEPTO, (prov, concepto),
                   LAG1_CONCEPTO, concepto, MEDIANA_GLOBAL_LOG)


def obtener_lag2(proveedor: str, concepto: str) -> float:
    prov = limpiar(proveedor)
    return _lookup(LAG2_PROV_CONCEPTO, (prov, concepto),
                   LAG1_CONCEPTO, concepto, MEDIANA_GLOBAL_LOG)


def obtener_lag3(proveedor: str, concepto: str) -> float:
    prov = limpiar(proveedor)
    return _lookup(LAG3_PROV_CONCEPTO, (prov, concepto),
                   LAG1_CONCEPTO, concepto, MEDIANA_GLOBAL_LOG)


def obtener_rolling3(proveedor: str, concepto: str) -> float:
    prov = limpiar(proveedor)
    return _lookup(ROLLING3_PROV_CONCEPTO, (prov, concepto),
                   LAG1_CONCEPTO, concepto, MEDIANA_GLOBAL_LOG)


def obtener_n_obs(proveedor: str, concepto: str) -> int:
    prov = limpiar(proveedor)
    return N_OBS_PROV_CONCEPTO.get((prov, concepto), 0)


def obtener_frecuencia_proveedor(proveedor: str) -> int:
    return FRECUENCIA_PROVEEDOR.get(limpiar(proveedor), 0)


def obtener_tarifa_por_contenedor(proveedor: str, concepto: str) -> float:
    prov = limpiar(proveedor)
    return _lookup(TARIFA_POR_CONTENEDOR_MAP, (prov, concepto),
                   TARIFA_POR_CONTENEDOR_CONCEPTO, concepto, TARIFA_GLOBAL)

# --- celda 4 ---
# Construcción de features para un despacho nuevo

def construir_features(despacho: dict) -> pd.DataFrame:
    """Construye un DataFrame con una fila por concepto canónico."""
    eta = pd.to_datetime(despacho.get("fecha_eta", pd.Timestamp.now()), errors="coerce")
    if pd.isna(eta):
        eta = pd.Timestamp.now()

    mes      = eta.month
    semana   = eta.isocalendar()[1]
    mode     = limpiar(despacho.get("mode", "FCL"))
    incoterm = limpiar(despacho.get("incoterm", "FOB"))
    pol      = limpiar(despacho.get("pol", "DESCONOCIDO"))
    proveedor    = limpiar(despacho.get("proveedor_principal", "DESCONOCIDO"))
    agencia      = limpiar(despacho.get("agencia_aduana", "DESCONOCIDO"))
    contenedores = float(despacho.get("contenedores", 1) or 1)
    bultos       = float(despacho.get("bultos", 0) or 0)
    peso_bruto   = float(despacho.get("peso_bruto_kg", 0) or 0)

    incoterm_grupo  = mapear_incoterm_grupo(incoterm)
    ruta_origen     = mapear_ruta_origen(pol)
    dias_transito   = estimar_dias_transito(mode)
    densidad_bultos = (bultos / contenedores) if contenedores > 0 else 0.0
    carga_peso      = (peso_bruto / contenedores) if contenedores > 0 else 0.0
    frecuencia_prov = obtener_frecuencia_proveedor(proveedor)
    log_contenedores = float(np.log1p(max(contenedores, 0)))
    log_bultos       = float(np.log1p(max(bultos, 0)))
    log_peso_bruto   = float(np.log1p(max(peso_bruto, 0)))

    filas = []
    for concepto in CONCEPTOS:
        tarifa        = obtener_tarifa_historica(proveedor, concepto)
        lag1          = obtener_lag1(proveedor, concepto)
        lag2          = obtener_lag2(proveedor, concepto)
        lag3          = obtener_lag3(proveedor, concepto)
        rolling3      = obtener_rolling3(proveedor, concepto)
        n_obs         = obtener_n_obs(proveedor, concepto)
        tarifa_x_ctnr = obtener_tarifa_por_contenedor(proveedor, concepto)

        te_prov   = TE_PROVEEDOR.get(proveedor, _TE_PROV_GLOBAL)
        te_ci_key = f"{concepto}_{incoterm_grupo}"
        te_cr_key = f"{concepto}_{ruta_origen}"
        te_ci     = TE_CONCEPTO_INCOTERM.get(te_ci_key, _TE_CI_GLOBAL)
        te_cr     = TE_CONCEPTO_RUTA.get(te_cr_key, _TE_CR_GLOBAL)
        med_cm    = MEDIANA_CONCEPTO_MODE.get(
            (concepto, mode), MEDIANA_POR_CONCEPTO.get(concepto, MEDIANA_GLOBAL_LOG)
        )
        med_conc  = MEDIANA_POR_CONCEPTO.get(concepto, MEDIANA_GLOBAL_LOG)
        diff_tar  = tarifa - med_conc
        prem_idx  = lag1 - med_conc
        cv_conc   = CONCEPTO_CV.get(concepto, 0.0)

        filas.append({
            "año":                  eta.year,
            "mes":                  mes,
            "trimestre":            (mes - 1) // 3 + 1,
            "semana_año":           semana,
            "dia_semana":           eta.dayofweek,
            "dias_desde_inicio":    max((eta - FECHA_INICIO).days, 0),
            "mes_sin":              np.sin(2 * np.pi * mes / 12),
            "mes_cos":              np.cos(2 * np.pi * mes / 12),
            "semana_sin":           np.sin(2 * np.pi * semana / 52),
            "semana_cos":           np.cos(2 * np.pi * semana / 52),
            "es_temporada_alta":    int(8 <= mes <= 12),
            "es_cierre_fiscal":     int(mes in [11, 12]),
            "dias_transito":        dias_transito,
            "densidad_bultos":      densidad_bultos,
            "carga_peso":           carga_peso,
            "tiene_proyecto":       int(bool(despacho.get("tiene_proyecto", False))),
            "tarifa_historica":     tarifa,
            "lag1_costo":           lag1,
            "lag2_costo":           lag2,
            "lag3_costo":           lag3,
            "rolling_median_3":     rolling3,
            "frecuencia_proveedor": frecuencia_prov,
            "te_concepto_incoterm": te_ci,
            "te_concepto_ruta":     te_cr,
            "te_proveedor":         te_prov,
            "median_concepto_mode": med_cm,
            "diff_tarifa_mediana":  diff_tar,
            "provider_premium_idx": prem_idx,
            "n_obs_prov_concepto":  n_obs,
            "concepto_cv":          cv_conc,
            "log_contenedores":     log_contenedores,
            "log_bultos":           log_bultos,
            "log_peso_bruto":       log_peso_bruto,
            "tarifa_por_contenedor": tarifa_x_ctnr,
            "concepto_canonico":    concepto,
            "incoterm_grupo":       incoterm_grupo,
            "ruta_origen":          ruta_origen,
            "proveedor_norm":       proveedor,
            "agencia_aduana_norm":  agencia,
            "mode":                 mode,
            "type":                 limpiar(despacho.get("type", "FCL")),
        })

    df_features = pd.DataFrame(filas)

    for col in FEATURES_NUM:
        if col not in df_features.columns:
            df_features[col] = regresion_medians.get(col, 0.0)
    for col in FEATURES_CAT:
        if col not in df_features.columns:
            df_features[col] = "DESCONOCIDO"
        df_features[col] = df_features[col].astype("category")

    return df_features

# --- celda 5 ---
# Cálculo de riesgo (usa umbrales absolutos del train, consistente con 03b)

def calcular_riesgo(X_fila: pd.DataFrame):
    cat_cols = [c for c in FEATURES_CAT if c in X_fila.columns]
    num_cols = [c for c in FEATURES_NUM if c in X_fila.columns]
    X_num = X_fila[num_cols].fillna(0).values
    X_cat = riesgo_encoder.transform(X_fila[cat_cols].astype(str))
    X_if  = np.hstack([X_num, X_cat])

    raw_score  = -float(iso_forest.decision_function(X_if).ravel()[0])
    score_min  = riesgo_score_params["min"]
    score_max  = riesgo_score_params["max"]
    score_norm = float(np.clip(
        (raw_score - score_min) / (score_max - score_min + 1e-9) * 100, 0, 100
    ))
    if score_norm >= 80:
        nivel = "ALTO"
    elif score_norm >= 50:
        nivel = "MEDIO"
    else:
        nivel = "BAJO"
    return score_norm, nivel

# --- celda 6 ---
# Aplicación de bias correction (misma lógica que 03a_regresion.py)

def aplicar_bias(pred_pen: float, concepto: str, proveedor=None) -> float:
    """Capa 1: piecewise por concepto. Capa 2: ratio por proveedor×concepto."""
    corr = bias_piecewise.get(concepto, bias_escalar.get(concepto, 1.0))
    if isinstance(corr, tuple):
        edges, ratios = corr
        bucket  = int(np.clip(np.searchsorted(edges[1:-1], pred_pen), 0, len(ratios) - 1))
        pred_l1 = pred_pen * ratios[bucket]
    else:
        pred_l1 = pred_pen * corr
    if proveedor is not None:
        ratio_l2 = bias_prov_concepto.get((str(proveedor).upper(), concepto), 1.0)
        return pred_l1 * ratio_l2
    return pred_l1


def predecir_despacho(despacho: dict) -> pd.DataFrame:
    df_feat   = construir_features(despacho)
    resultados = []

    for concepto in CONCEPTOS:
        if concepto not in modelos_por_concepto:
            continue
        fila = df_feat[df_feat["concepto_canonico"] == concepto]
        if len(fila) == 0:
            continue

        X_pred    = fila[[c for c in ALL_FEATURES if c in fila.columns]]
        proveedor = fila["proveedor_norm"].iloc[0] if "proveedor_norm" in fila.columns else None

        pred_log  = float(modelos_por_concepto[concepto].predict(X_pred)[0])
        pred_pen  = float(np.expm1(pred_log))
        pred_corr = aplicar_bias(pred_pen, concepto, proveedor=proveedor)

        p10_raw  = float(np.expm1(modelos_p10[concepto].predict(X_pred)[0]))
        p90_raw  = float(np.expm1(modelos_p90[concepto].predict(X_pred)[0]))
        margenes = cqr_margenes.get(concepto, {"margen_inf": 0.20, "margen_sup": 0.20})
        p10_cal  = max(p10_raw * (1 - margenes["margen_inf"]), 0)
        p90_cal  = p90_raw * (1 + margenes["margen_sup"])

        riesgo_score, nivel_riesgo = calcular_riesgo(fila)

        resultados.append({
            "concepto":           concepto,
            "costo_estimado_pen": round(pred_corr, 2),
            "p10":                round(p10_cal, 2),
            "p90":                round(p90_cal, 2),
            "nivel_riesgo":       nivel_riesgo,
            "riesgo_score":       round(riesgo_score, 1),
        })

    return pd.DataFrame(resultados)

# --- celda 7 ---
# Presentación de resultados

def imprimir_resultado(despacho: dict, df_res: pd.DataFrame):
    total     = df_res["costo_estimado_pen"].sum()
    p10_total = df_res["p10"].sum()
    p90_total = df_res["p90"].sum()
    amplitud  = (p90_total - p10_total) / total * 100 if total > 0 else 0.0

    print("\n" + "=" * 70)
    print("  PREDICCIÓN DE COSTOS DE IMPORTACIÓN")
    print("=" * 70)
    print(f"  Proveedor    : {despacho.get('proveedor_principal', '-')}")
    print(f"  Puerto origen: {despacho.get('pol', '-')} → Callao")
    print(f"  Modalidad    : {despacho.get('mode', '-')} / {despacho.get('type', '-')}")
    print(f"  Incoterm     : {despacho.get('incoterm', '-')}")
    print(f"  Contenedores : {despacho.get('contenedores', 1)}")
    print(f"  Peso bruto   : {despacho.get('peso_bruto_kg', 0):,} kg")
    print(f"  ETA estimada : {despacho.get('fecha_eta', '-')}")
    print("-" * 70)
    print(f"  {'CONCEPTO':<30} {'ESTIMADO':>12} {'P10':>12} {'P90':>12}  RIESGO")
    print("-" * 70)

    for _, row in df_res.sort_values("costo_estimado_pen", ascending=False).iterrows():
        alerta = " ⚠" if row["nivel_riesgo"] == "ALTO" else ""
        print(
            f"  {row['concepto']:<30} "
            f"S/{row['costo_estimado_pen']:>10,.0f} "
            f"S/{row['p10']:>10,.0f} "
            f"S/{row['p90']:>10,.0f}"
            f"  {row['nivel_riesgo']}{alerta}"
        )

    print("=" * 70)
    print(f"  {'TOTAL ESTIMADO':<30} S/{total:>10,.0f} S/{p10_total:>10,.0f} S/{p90_total:>10,.0f}")
    print("=" * 70)

    calidad = "PRECISO" if amplitud < 30 else ("MODERADO" if amplitud < 60 else "ANCHO")
    print(f"\n  Amplitud del intervalo: {amplitud:.1f}%  → {calidad}")

    alertas = df_res[df_res["nivel_riesgo"] == "ALTO"]
    if len(alertas) > 0:
        print(f"\n  ALERTAS — {len(alertas)} concepto(s) con RIESGO ALTO:")
        for _, row in alertas.iterrows():
            print(f"    • {row['concepto']} (score={row['riesgo_score']:.0f}/100)")

# --- celda 8 ---
# Ejemplo 1: FCL marítimo desde Valencia
print("\n" + "#" * 70)
print("# EJEMPLO 1: Despacho FCL marítimo  (Valencia → Callao)           #")
print("#" * 70)

despacho_1 = {
    "proveedor_principal": "HORTIFRUT CHILE",
    "pol":                 "VALENCIA",
    "mode":                "FCL",
    "type":                "40HC",
    "incoterm":            "FOB",
    "contenedores":        2,
    "bultos":              480,
    "peso_bruto_kg":       18_000,
    "fecha_eta":           "2025-09-15",
    "tiene_proyecto":      True,
    "agencia_aduana":      "AVM ADUANERA",
}
df_pred_1 = predecir_despacho(despacho_1)
imprimir_resultado(despacho_1, df_pred_1)

# --- celda 9 ---
# Ejemplo 2: Despacho aéreo desde Santiago
print("\n\n" + "#" * 70)
print("# EJEMPLO 2: Despacho aéreo  (Santiago → Lima)                    #")
print("#" * 70)

despacho_2 = {
    "proveedor_principal": "AGROCHIMICA CODIAGRO",
    "pol":                 "SANTIAGO",
    "mode":                "AIR",
    "type":                "AIR CARGO",
    "incoterm":            "CPT",
    "contenedores":        0,
    "bultos":              12,
    "peso_bruto_kg":       500,
    "fecha_eta":           "2025-11-03",
    "tiene_proyecto":      False,
    "agencia_aduana":      "BOXPOOL",
}
df_pred_2 = predecir_despacho(despacho_2)
imprimir_resultado(despacho_2, df_pred_2)

# --- celda 10 ---
# Validación masiva en el test set + gráfico de residuos
# El test set se lee UNA SOLA VEZ y se reutiliza en la muestra aleatoria.
print("\n\n" + "=" * 70)
print("VALIDACIÓN EN TEST SET COMPLETO")
print("=" * 70)

test_path = PROCESSED_DIR / "dataset_modelable_test.parquet"
if not test_path.exists():
    print("Test set no encontrado. Ejecuta primero 02_feature_engineering.py")
else:
    test_val = pd.read_parquet(test_path)
    test_val["log_target"] = np.log1p(test_val["importe_total_pen"])
    col_prov_te = COL_PROV if COL_PROV in test_val.columns else "proveedor"
    cols_disp   = [c for c in ALL_FEATURES if c in test_val.columns]

    errores_por_concepto   = []
    cobertura_por_concepto = []
    todos_reales           = []   # para el gráfico de residuos
    todos_predichos        = []

    for concepto, modelo in modelos_por_concepto.items():
        mask = test_val["concepto_canonico"] == concepto
        if mask.sum() < 5:
            continue

        X_te = test_val.loc[mask, cols_disp].copy()
        for col in FEATURES_CAT:
            if col in X_te.columns:
                X_te[col] = X_te[col].astype("category")

        y_te      = test_val.loc[mask, "importe_total_pen"].values
        provs_te  = (test_val.loc[mask, col_prov_te].values
                     if col_prov_te in test_val.columns else [""] * len(y_te))
        pred_log  = modelo.predict(X_te)
        pred_pen  = np.expm1(pred_log)
        pred_corr = np.array([
            aplicar_bias(p, concepto, prov) for p, prov in zip(pred_pen, provs_te)
        ])

        mape = mediana_porcentaje_error_absoluto(y_te, pred_corr)   # función de utils

        p10_te  = np.expm1(modelos_p10[concepto].predict(X_te))
        p90_te  = np.expm1(modelos_p90[concepto].predict(X_te))
        marg    = cqr_margenes.get(concepto, {"margen_inf": 0.2, "margen_sup": 0.2})
        p10_cal = p10_te * (1 - marg["margen_inf"])
        p90_cal = p90_te * (1 + marg["margen_sup"])
        cobertura = ((y_te >= p10_cal) & (y_te <= p90_cal)).mean() * 100

        errores_por_concepto.append({"concepto": concepto, "n": int(mask.sum()), "MdAPE_%": mape})
        cobertura_por_concepto.append({"concepto": concepto, "Cobertura_%": cobertura})
        todos_reales.extend(y_te.tolist())
        todos_predichos.extend(pred_corr.tolist())

    df_eval = (
        pd.DataFrame(errores_por_concepto)
        .merge(pd.DataFrame(cobertura_por_concepto), on="concepto")
        .set_index("concepto")
    )

    print(f"\n{'CONCEPTO':<30} {'N':>6} {'MdAPE%':>8} {'Cobertura%':>12}")
    print("-" * 60)
    for concepto, row in df_eval.sort_values("MdAPE_%").iterrows():
        objetivo_ok = "✓" if row["MdAPE_%"] < 5 else " "
        print(f"{concepto:<30} {int(row['n']):>6} {row['MdAPE_%']:>8.1f}"
              f" {row['Cobertura_%']:>12.1f}  {objetivo_ok}")

    mdape_global  = df_eval["MdAPE_%"].mean()
    cobert_global = df_eval["Cobertura_%"].mean()
    print("=" * 60)
    print(f"{'PROMEDIO':<30} {int(df_eval['n'].mean()):>6}"
          f" {mdape_global:>8.1f} {cobert_global:>12.1f}")
    print(f"\nMdAPE global          : {mdape_global:.1f}%")
    print(f"Cobertura P10-P90     : {cobert_global:.1f}%")
    print(f"Objetivo MdAPE < 5%   : "
          f"{'ALCANZADO ✓' if mdape_global < 5 else f'EN PROGRESO ({mdape_global:.1f}%)'}")

    # Gráfico de residuos — detecta sesgo sistemático por rango de precio
    todos_reales    = np.array(todos_reales)
    todos_predichos = np.array(todos_predichos)
    residuos_pct    = (todos_reales - todos_predichos) / np.clip(todos_reales, 1, None) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(np.log1p(todos_predichos), residuos_pct,
                    alpha=0.3, s=8, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("log1p(Predicho PEN)")
    axes[0].set_ylabel("Residuo %  [(real − pred) / real × 100]")
    axes[0].set_title("Residuos vs. Predicho (test set)")

    mediana_res = float(np.median(residuos_pct))
    axes[1].hist(residuos_pct, bins=50, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1)
    axes[1].axvline(mediana_res, color="orange", linewidth=1.5, linestyle="--",
                    label=f"Mediana={mediana_res:.1f}%")
    axes[1].set_xlabel("Residuo %")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos (test)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demo_residuos.png", dpi=120)
    plt.close()
    print(f"\nGráfico de residuos guardado: reports/figures/demo_residuos.png")

    # --- celda 11 ---
    # Muestra aleatoria de 10 registros reales del test (reutiliza test_val ya cargado)
    print("\n\n" + "#" * 82)
    print("# MUESTRA ALEATORIA — 10 DATOS REALES DEL TEST CON PRECIO PREDICHO           #")
    print("#" * 82)

    test_muestra = test_val[test_val["concepto_canonico"].isin(CONCEPTOS)].copy()
    muestra      = (test_muestra
                    .sample(n=min(10, len(test_muestra)), random_state=42)
                    .reset_index(drop=True))

    print(f"\n{'N°':>3}  {'CONCEPTO':<28} {'PROVEEDOR':<22}"
          f" {'REAL':>12} {'PREDICHO':>14} {'P10':>11} {'P90':>11}  {'ERR%':>7}")
    print("-" * 110)

    errores_muestra  = []
    dentro_intervalo = 0
    col_prov_label   = COL_PROV if COL_PROV in muestra.columns else "proveedor"

    for idx, (_, row) in enumerate(muestra.iterrows(), start=1):
        concepto = row["concepto_canonico"]
        modelo   = modelos_por_concepto[concepto]

        X_fila = pd.DataFrame([row[cols_disp]])
        for col in FEATURES_CAT:
            if col in X_fila.columns:
                X_fila[col] = X_fila[col].astype("category")

        pred_log  = float(modelo.predict(X_fila)[0])
        pred_pen  = float(np.expm1(pred_log))
        prov_fila = str(row.get(col_prov_label, ""))
        pred_corr = aplicar_bias(pred_pen, concepto, proveedor=prov_fila)

        p10_raw  = float(np.expm1(modelos_p10[concepto].predict(X_fila)[0]))
        p90_raw  = float(np.expm1(modelos_p90[concepto].predict(X_fila)[0]))
        marg     = cqr_margenes.get(concepto, {"margen_inf": 0.20, "margen_sup": 0.20})
        p10_cal  = max(p10_raw * (1 - marg["margen_inf"]), 0)
        p90_cal  = p90_raw * (1 + marg["margen_sup"])

        real      = float(row["importe_total_pen"])
        error_pct = abs(real - pred_corr) / max(real, 1) * 100
        en_rango  = p10_cal <= real <= p90_cal
        dentro    = "✓" if en_rango else " "
        errores_muestra.append(error_pct)
        if en_rango:
            dentro_intervalo += 1

        proveedor_str = str(row.get(col_prov_label, "-"))[:20]
        print(
            f"{idx:>3}. {concepto:<28} {proveedor_str:<22}"
            f" S/{real:>10,.0f} S/{pred_corr:>12,.0f}"
            f" S/{p10_cal:>9,.0f} S/{p90_cal:>9,.0f}"
            f"  {error_pct:>6.1f}% {dentro}"
        )

    n_muestra = len(muestra)
    print("=" * 110)
    print(f"  Error promedio (MAPE)  : {np.mean(errores_muestra):.1f}%")
    print(f"  Error mediano (MdAPE)  : {np.median(errores_muestra):.1f}%")
    print(f"  Dentro del intervalo   : {dentro_intervalo}/{n_muestra}"
          f"  (✓ = real dentro de P10-P90)")
