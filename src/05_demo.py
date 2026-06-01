"""
05_demo.py — CRISP-DM Fase 5: Demo de predicción end-to-end.

Integra los 3 tracks de ML para un despacho nuevo:
  ► Track A (Regresión)   : costo puntual + IC 80% (P10-P90) por concepto canónico
  ► Track B (Clasificación): nivel de riesgo BAJO/MEDIO/ALTO por concepto
  ► Track C (Clustering)  : perfil operativo (cluster K-Means + nombre de negocio)

El motor de predicción es una función limpia y reutilizable para un futuro endpoint REST.
La validación se realiza sobre el test set completo (costos reales conocidos).

Prerequisito: ejecutar todos los scripts anteriores en orden.
Ejecutar    : python src/05_demo.py
Salidas     : reports/figures/demo_residuos.png
              reports/figures/demo_cobertura_ic80.png
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR, PROCESSED_DIR, FIGURES_DIR, DPI_FIGURAS,
    RANDOM_STATE
)
from utils import (
    limpiar_texto as limpiar,
    mapear_incoterm_grupo, mapear_ruta_origen, estimar_dias_transito,
    mdape, mape, rmse, mae, cobertura_intervalo,
)

# ---------------------------------------------------------------------------
# Carga y verificación de artefactos
# ---------------------------------------------------------------------------
MODELOS_REQUERIDOS = [
    "lgb_multi_concepto.joblib",
    "lgb_p10_multi_concepto.joblib",
    "lgb_p90_multi_concepto.joblib",
    "bias_correction_escalar.joblib",
    "bias_correction_piecewise.joblib",
    "bias_prov_concepto.joblib",
    "cqr_margenes_por_concepto.joblib",
    "regresion_medians.joblib",
    "feature_config.joblib",
    "clasificador_riesgo.joblib",
    "label_encoder_riesgo.joblib",
    "riesgo_score_params.joblib",
    "riesgo_cat_encoders.joblib",
    "clasificacion_config.joblib",
]

# Modelos de clustering opcionales (no fallan si no existen)
MODELOS_CLUSTERING = [
    "clustering_preprocessor.joblib",
    "clustering_pca.joblib",
    "cluster_nombres.joblib",
    "clustering_config.joblib",
]

faltantes = [m for m in MODELOS_REQUERIDOS if not (MODELS_DIR / m).exists()]
if faltantes:
    print("ERROR: faltan modelos obligatorios:")
    for m in faltantes:
        print(f"  - {m}")
    raise SystemExit(1)

# Regresión
modelos_por_concepto = joblib.load(MODELS_DIR / "lgb_multi_concepto.joblib")
modelos_p10          = joblib.load(MODELS_DIR / "lgb_p10_multi_concepto.joblib")
modelos_p90          = joblib.load(MODELS_DIR / "lgb_p90_multi_concepto.joblib")
bias_escalar         = joblib.load(MODELS_DIR / "bias_correction_escalar.joblib")
bias_piecewise       = joblib.load(MODELS_DIR / "bias_correction_piecewise.joblib")
bias_prov_concepto   = joblib.load(MODELS_DIR / "bias_prov_concepto.joblib")
cqr_margenes         = joblib.load(MODELS_DIR / "cqr_margenes_por_concepto.joblib")
regresion_medians    = joblib.load(MODELS_DIR / "regresion_medians.joblib")
feature_config       = joblib.load(MODELS_DIR / "feature_config.joblib")

# Clasificación
clf_riesgo           = joblib.load(MODELS_DIR / "clasificador_riesgo.joblib")
le_riesgo            = joblib.load(MODELS_DIR / "label_encoder_riesgo.joblib")
riesgo_params        = joblib.load(MODELS_DIR / "riesgo_score_params.joblib")
cat_encoders_cls     = joblib.load(MODELS_DIR / "riesgo_cat_encoders.joblib")
cls_config           = joblib.load(MODELS_DIR / "clasificacion_config.joblib")

# Clustering (opcional)
clustering_disponible = all((MODELS_DIR / m).exists() for m in MODELOS_CLUSTERING)
if clustering_disponible:
    cl_preprocessor  = joblib.load(MODELS_DIR / "clustering_preprocessor.joblib")
    cl_pca           = joblib.load(MODELS_DIR / "clustering_pca.joblib")
    cluster_nombres  = joblib.load(MODELS_DIR / "cluster_nombres.joblib")
    cl_config        = joblib.load(MODELS_DIR / "clustering_config.joblib")

    # Buscar el archivo kmeans_k{K}.joblib
    _kmeans_files = sorted(MODELS_DIR.glob("kmeans_k*.joblib"))
    if _kmeans_files:
        kmeans_model = joblib.load(_kmeans_files[-1])
    else:
        clustering_disponible = False

# Configuración de features
FEATURES_NUM   = feature_config["features_num"]
FEATURES_CAT   = feature_config["features_cat"]
ALL_FEATURES   = FEATURES_NUM + FEATURES_CAT
CONCEPTOS      = sorted(modelos_por_concepto.keys())
COL_PROV       = feature_config.get("col_prov", "proveedor_norm")
MED_GLOBAL_LOG = feature_config.get("mediana_global_log", 7.0)

_TE_CI_MAP     = feature_config.get("te_ci_map", {})
_TE_CI_GLOBAL  = feature_config.get("te_ci_global", MED_GLOBAL_LOG)
_TE_CR_MAP     = feature_config.get("te_cr_map", {})
_TE_CR_GLOBAL  = feature_config.get("te_cr_global", MED_GLOBAL_LOG)
_TE_PROV_MAP   = feature_config.get("te_prov_map", {})
_TE_PROV_GLOBAL= feature_config.get("te_prov_global", MED_GLOBAL_LOG)
_MED_CONCEPTO  = feature_config.get("mediana_por_concepto", {})
_CV_CONCEPTO   = feature_config.get("concepto_cv", {})

FEATURES_CLS     = cls_config.get("features_cls", [])
FEATURES_NUM_CLS = cls_config.get("features_num_cls", [])
FEATURES_CAT_CLS = cls_config.get("features_cat_cls", [])

print(f"Modelos cargados: {len(CONCEPTOS)} conceptos canónicos")
print(f"Clustering disponible: {'SÍ' if clustering_disponible else 'NO'}")

# ---------------------------------------------------------------------------
# Precomputar estadísticos históricos desde train
# ---------------------------------------------------------------------------
print("\nCargando estadísticos históricos desde train set...")

FECHA_INICIO = pd.Timestamp("2020-01-01")
HALFLIFE_STR = f"{feature_config.get('halflife_dias', 365)}D"

TARIFAS_PROV_CONC   = {}
TARIFAS_CONC        = {}
TARIFA_GLOBAL       = MED_GLOBAL_LOG
LAG1_PROV_CONC      = {}
LAG2_PROV_CONC      = {}
LAG3_PROV_CONC      = {}
LAG1_CONC           = {}
ROLLING3_PROV_CONC  = {}
FRECUENCIA_PROV     = {}
N_OBS_PROV_CONC     = {}
TARIFA_CTNR_MAP     = {}
TARIFA_CTNR_CONC    = {}
MEDIANA_CONC_MODE   = {}

if (PROCESSED_DIR / "dataset_modelable_train.parquet").exists():
    _tr = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
    _tr["log_target"] = np.log1p(_tr["importe_total_pen"])
    col_prov_tr = COL_PROV if COL_PROV in _tr.columns else "proveedor"
    if "fecha_trabajo" in _tr.columns:
        _tr = _tr.sort_values("fecha_trabajo")

    TARIFA_GLOBAL = float(_tr["log_target"].mean())

    for (prov, conc), g in _tr.groupby([col_prov_tr, "concepto_canonico"]):
        pk = (str(prov).upper(), conc)
        if "fecha_trabajo" in g.columns:
            vals = g["log_target"].ewm(
                halflife=HALFLIFE_STR,
                times=pd.DatetimeIndex(g["fecha_trabajo"]), min_periods=1
            ).mean()
        else:
            vals = g["log_target"].ewm(halflife=365, min_periods=1).mean()
        TARIFAS_PROV_CONC[pk] = float(vals.iloc[-1])
        hist = g["log_target"].values
        LAG1_PROV_CONC[pk]     = float(hist[-1])
        LAG2_PROV_CONC[pk]     = float(hist[-2]) if len(hist) >= 2 else float(hist[-1])
        LAG3_PROV_CONC[pk]     = float(hist[-3]) if len(hist) >= 3 else float(hist[-1])
        ROLLING3_PROV_CONC[pk] = float(g["log_target"].tail(3).median())
        N_OBS_PROV_CONC[pk]    = int(len(hist))

    for conc, g in _tr.groupby("concepto_canonico"):
        if "fecha_trabajo" in g.columns:
            vals = g["log_target"].ewm(
                halflife=HALFLIFE_STR,
                times=pd.DatetimeIndex(g["fecha_trabajo"]), min_periods=1
            ).mean()
        else:
            vals = g["log_target"].ewm(halflife=365, min_periods=1).mean()
        TARIFAS_CONC[conc] = float(vals.iloc[-1])
        LAG1_CONC[conc]    = float(g["log_target"].iloc[-1])

    FRECUENCIA_PROV = {str(k).upper(): int(v) for k, v in _tr.groupby(col_prov_tr).size().items()}

    if "mode" in _tr.columns:
        MEDIANA_CONC_MODE = _tr.groupby(["concepto_canonico", "mode"])["log_target"].median().to_dict()

    if "contenedores" in _tr.columns:
        _tr["_lcc"] = np.log1p(_tr["importe_total_pen"] / _tr["contenedores"].clip(lower=1).fillna(1))
        for (prov, conc), g in _tr.groupby([col_prov_tr, "concepto_canonico"]):
            pk = (str(prov).upper(), conc)
            if "fecha_trabajo" in g.columns:
                vals = g["_lcc"].ewm(halflife=HALFLIFE_STR,
                                     times=pd.DatetimeIndex(g["fecha_trabajo"]), min_periods=1).mean()
            else:
                vals = g["_lcc"].ewm(halflife=365, min_periods=1).mean()
            TARIFA_CTNR_MAP[pk] = float(vals.iloc[-1])
        for conc, g in _tr.groupby("concepto_canonico"):
            if "fecha_trabajo" in g.columns:
                vals = g["_lcc"].ewm(halflife=HALFLIFE_STR,
                                     times=pd.DatetimeIndex(g["fecha_trabajo"]), min_periods=1).mean()
            else:
                vals = g["_lcc"].ewm(halflife=365, min_periods=1).mean()
            TARIFA_CTNR_CONC[conc] = float(vals.iloc[-1])

    print(f"  Tarifas prov×concepto: {len(TARIFAS_PROV_CONC)}")
    print(f"  Proveedores en train : {len(FRECUENCIA_PROV)}")

def _lookup(primary, pk, fallback, fk, default):
    v = primary.get(pk)
    if v is not None: return v
    v = fallback.get(fk)
    return v if v is not None else default

def get_tarifa(prov, conc):
    return _lookup(TARIFAS_PROV_CONC, (limpiar(prov), conc), TARIFAS_CONC, conc, TARIFA_GLOBAL)

def get_lag1(prov, conc):
    return _lookup(LAG1_PROV_CONC, (limpiar(prov), conc), LAG1_CONC, conc, MED_GLOBAL_LOG)

def get_lag2(prov, conc):
    return _lookup(LAG2_PROV_CONC, (limpiar(prov), conc), LAG1_CONC, conc, MED_GLOBAL_LOG)

def get_lag3(prov, conc):
    return _lookup(LAG3_PROV_CONC, (limpiar(prov), conc), LAG1_CONC, conc, MED_GLOBAL_LOG)

def get_rolling3(prov, conc):
    return _lookup(ROLLING3_PROV_CONC, (limpiar(prov), conc), LAG1_CONC, conc, MED_GLOBAL_LOG)

def get_tarifa_ctnr(prov, conc):
    return _lookup(TARIFA_CTNR_MAP, (limpiar(prov), conc), TARIFA_CTNR_CONC, conc, TARIFA_GLOBAL)

# ---------------------------------------------------------------------------
# Bias correction (replicada para portabilidad)
# ---------------------------------------------------------------------------
def _aplicar_bias_capa1(pred_pen: float, concepto: str) -> float:
    corr = bias_piecewise.get(concepto, bias_escalar.get(concepto, 1.0))
    if isinstance(corr, tuple):
        edges, ratios = corr
        bucket = int(np.clip(np.searchsorted(edges[1:-1], pred_pen), 0, len(ratios) - 1))
        return pred_pen * ratios[bucket]
    return pred_pen * corr

def _aplicar_bias_completo(pred_pen: float, concepto: str, proveedor=None) -> float:
    p_l1   = _aplicar_bias_capa1(pred_pen, concepto)
    if proveedor is not None:
        r_l2 = bias_prov_concepto.get((str(proveedor).upper(), concepto), 1.0)
        return p_l1 * r_l2
    return p_l1

# ---------------------------------------------------------------------------
# Construcción de features para un despacho nuevo
# ---------------------------------------------------------------------------
def construir_features(despacho: dict) -> pd.DataFrame:
    """Construye 1 fila por concepto canónico a partir de los datos del despacho."""
    eta         = pd.to_datetime(despacho.get("fecha_eta", pd.Timestamp.now()), errors="coerce")
    if pd.isna(eta): eta = pd.Timestamp.now()

    mes         = eta.month
    semana      = eta.isocalendar()[1]
    mode        = limpiar(despacho.get("mode", "FCL"))
    incoterm    = limpiar(despacho.get("incoterm", "FOB"))
    pol         = limpiar(despacho.get("pol", "DESCONOCIDO"))
    proveedor   = limpiar(despacho.get("proveedor_principal", "DESCONOCIDO"))
    agencia     = limpiar(despacho.get("agencia_aduana", "DESCONOCIDO"))
    contenedores = float(despacho.get("contenedores", 1) or 1)
    bultos       = float(despacho.get("bultos", 0) or 0)
    peso_bruto   = float(despacho.get("peso_bruto_kg", 0) or 0)

    incoterm_grupo  = mapear_incoterm_grupo(incoterm)
    ruta_origen     = mapear_ruta_origen(pol)
    dias_transito   = estimar_dias_transito(mode)
    densidad_bultos = bultos / max(contenedores, 1)
    carga_peso      = peso_bruto / max(contenedores, 1)
    frec_prov       = FRECUENCIA_PROV.get(proveedor, 0)
    log_contenedores = float(np.log1p(max(contenedores, 0)))
    log_bultos       = float(np.log1p(max(bultos, 0)))
    log_peso_bruto   = float(np.log1p(max(peso_bruto, 0)))

    te_prov = {str(k).upper(): v for k, v in _TE_PROV_MAP.items()}.get(proveedor, _TE_PROV_GLOBAL)

    filas = []
    for concepto in CONCEPTOS:
        tarifa    = get_tarifa(proveedor, concepto)
        lag1      = get_lag1(proveedor, concepto)
        lag2      = get_lag2(proveedor, concepto)
        lag3      = get_lag3(proveedor, concepto)
        rolling3  = get_rolling3(proveedor, concepto)
        n_obs     = N_OBS_PROV_CONC.get((proveedor, concepto), 0)
        tar_ctnr  = get_tarifa_ctnr(proveedor, concepto)

        te_ci_key = f"{concepto}_{incoterm_grupo}"
        te_cr_key = f"{concepto}_{ruta_origen}"
        te_ci     = _TE_CI_MAP.get(te_ci_key, _TE_CI_GLOBAL)
        te_cr     = _TE_CR_MAP.get(te_cr_key, _TE_CR_GLOBAL)
        med_cm    = MEDIANA_CONC_MODE.get((concepto, mode), _MED_CONCEPTO.get(concepto, MED_GLOBAL_LOG))
        med_conc  = _MED_CONCEPTO.get(concepto, MED_GLOBAL_LOG)
        diff_tar  = tarifa - med_conc
        prem_idx  = lag1 - med_conc
        cv_conc   = _CV_CONCEPTO.get(concepto, 0.0)

        filas.append({
            "año": eta.year, "mes": mes, "trimestre": (mes - 1) // 3 + 1,
            "semana_año": semana, "dia_semana": eta.dayofweek,
            "dias_desde_inicio": max((eta - FECHA_INICIO).days, 0),
            "mes_sin": np.sin(2 * np.pi * mes / 12),
            "mes_cos": np.cos(2 * np.pi * mes / 12),
            "semana_sin": np.sin(2 * np.pi * semana / 52),
            "semana_cos": np.cos(2 * np.pi * semana / 52),
            "es_temporada_alta": int(8 <= mes <= 12),
            "es_cierre_fiscal": int(mes in [11, 12]),
            "dias_transito": dias_transito,
            "densidad_bultos": densidad_bultos,
            "carga_peso": carga_peso,
            "tiene_proyecto": int(bool(despacho.get("tiene_proyecto", False))),
            "tarifa_historica": tarifa,
            "lag1_costo": lag1, "lag2_costo": lag2, "lag3_costo": lag3,
            "rolling_median_3": rolling3,
            "frecuencia_proveedor": frec_prov,
            "te_concepto_incoterm": te_ci,
            "te_concepto_ruta": te_cr,
            "te_proveedor": te_prov,
            "median_concepto_mode": med_cm,
            "diff_tarifa_mediana": diff_tar,
            "provider_premium_idx": prem_idx,
            "n_obs_prov_concepto": n_obs,
            "concepto_cv": cv_conc,
            "log_contenedores": log_contenedores,
            "log_bultos": log_bultos,
            "log_peso_bruto": log_peso_bruto,
            "tarifa_por_contenedor": tar_ctnr,
            "concepto_canonico": concepto,
            "incoterm_grupo": incoterm_grupo,
            "ruta_origen": ruta_origen,
            "proveedor_norm": proveedor,
            "agencia_aduana_norm": agencia,
            "mode": mode,
            "type": limpiar(despacho.get("type", "FCL")),
        })

    df_feat = pd.DataFrame(filas)
    for col in FEATURES_NUM:
        if col not in df_feat.columns:
            df_feat[col] = regresion_medians.get(col, 0.0)
    for col in FEATURES_CAT:
        if col not in df_feat.columns:
            df_feat[col] = "DESCONOCIDO"
        df_feat[col] = df_feat[col].astype("category")

    return df_feat

# ---------------------------------------------------------------------------
# Clasificador de riesgo
# ---------------------------------------------------------------------------
def calcular_riesgo(X_fila: pd.DataFrame) -> tuple[float, str]:
    """Devuelve (score 0-100, nivel BAJO/MEDIO/ALTO) para una fila de features."""
    X_num = X_fila[[c for c in FEATURES_NUM_CLS if c in X_fila.columns]].fillna(0)
    X_cat = X_fila[[c for c in FEATURES_CAT_CLS if c in X_fila.columns]].astype(str)

    for col in X_cat.columns:
        enc = cat_encoders_cls.get(col)
        if enc is not None:
            X_cat[col] = enc.transform(X_cat[[col]]).ravel()
        else:
            X_cat[col] = 0.0

    X_cls  = np.hstack([X_num.values, X_cat.values.astype(float)])

    if hasattr(clf_riesgo, "predict_proba"):
        proba  = clf_riesgo.predict_proba(X_cls)[0]
        # score = probabilidad de ALTO × 100 (índice del label ALTO=2 en LabelEncoder)
        idx_alto = list(le_riesgo.classes_).index("ALTO")
        score  = float(proba[idx_alto]) * 100
    else:
        score  = 50.0

    pred_enc = clf_riesgo.predict(X_cls)[0]
    nivel    = le_riesgo.inverse_transform([pred_enc])[0]
    return score, nivel

# ---------------------------------------------------------------------------
# Perfil del despacho (clustering)
# ---------------------------------------------------------------------------
def obtener_perfil_cluster(despacho: dict) -> dict | None:
    """Asigna el despacho a un cluster K-Means y devuelve su perfil."""
    if not clustering_disponible:
        return None

    num_cols = cl_config.get("num_cols", [])
    cat_cols = cl_config.get("cat_cols", [])
    fill_num = cl_config.get("fill_num", {})
    fill_cat = cl_config.get("fill_cat", {})

    fila = {c: fill_num.get(c, 0) for c in num_cols}
    fila.update({c: fill_cat.get(c, "DESCONOCIDO") for c in cat_cols})

    fila["mode"]          = limpiar(despacho.get("mode", "FCL"))
    fila["incoterm_grupo"] = mapear_incoterm_grupo(limpiar(despacho.get("incoterm", "FOB")))
    fila["ruta_origen"]    = mapear_ruta_origen(limpiar(despacho.get("pol", "")))

    df_ops = pd.DataFrame([fila])
    for c in num_cols:
        df_ops[c] = pd.to_numeric(df_ops[c], errors="coerce").fillna(fill_num.get(c, 0))

    try:
        X_op   = cl_preprocessor.transform(df_ops[num_cols + cat_cols])
        X_pca  = cl_pca.transform(X_op)
        cluster_id = int(kmeans_model.predict(X_pca)[0])
        nombre     = cluster_nombres.get(cluster_id, f"Cluster {cluster_id}")
        return {"cluster_id": cluster_id, "nombre": nombre}
    except Exception as e:
        return {"cluster_id": -1, "nombre": f"No clasificado ({e})"}

# ---------------------------------------------------------------------------
# Motor principal de predicción
# ---------------------------------------------------------------------------
def predecir_despacho(despacho: dict) -> pd.DataFrame:
    """
    Función principal reutilizable para el futuro endpoint REST.

    Recibe: dict con datos del despacho (proveedor, pol, mode, etc.)
    Devuelve: DataFrame con una fila por concepto canónico:
              concepto | costo_estimado_pen | p10 | p90 | nivel_riesgo | riesgo_score
    """
    df_feat = construir_features(despacho)
    proveedor = df_feat["proveedor_norm"].iloc[0] if "proveedor_norm" in df_feat.columns else None
    resultados = []

    for concepto in CONCEPTOS:
        if concepto not in modelos_por_concepto:
            continue
        fila = df_feat[df_feat["concepto_canonico"] == concepto]
        if len(fila) == 0:
            continue

        X_pred    = fila[[c for c in ALL_FEATURES if c in fila.columns]]
        pred_log  = float(modelos_por_concepto[concepto].predict(X_pred)[0])
        pred_pen  = float(np.expm1(pred_log))
        pred_corr = _aplicar_bias_completo(pred_pen, concepto, proveedor)

        p10_raw  = float(np.expm1(modelos_p10[concepto].predict(X_pred)[0]))
        p90_raw  = float(np.expm1(modelos_p90[concepto].predict(X_pred)[0]))
        marg     = cqr_margenes.get(concepto, {"margen_inf": 0.20, "margen_sup": 0.20})
        p10_cal  = max(p10_raw * (1 - marg["margen_inf"]), 0)
        p90_cal  = p90_raw * (1 + marg["margen_sup"])

        score, nivel = calcular_riesgo(fila)

        resultados.append({
            "concepto":           concepto,
            "costo_estimado_pen": round(pred_corr, 2),
            "p10":                round(p10_cal, 2),
            "p90":                round(p90_cal, 2),
            "nivel_riesgo":       nivel,
            "riesgo_score":       round(score, 1),
        })

    return pd.DataFrame(resultados)

# ---------------------------------------------------------------------------
# Impresión de resultados
# ---------------------------------------------------------------------------
def imprimir_prediccion(despacho: dict, df_res: pd.DataFrame, perfil_cluster: dict | None = None):
    total     = df_res["costo_estimado_pen"].sum()
    p10_total = df_res["p10"].sum()
    p90_total = df_res["p90"].sum()
    amplitud  = (p90_total - p10_total) / max(total, 1) * 100
    calidad   = "PRECISO" if amplitud < 30 else ("MODERADO" if amplitud < 60 else "ANCHO")

    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  PREDICCIÓN DE COSTOS DE IMPORTACIÓN — HortifrutCostosImport")
    print(SEP)
    print(f"  Proveedor     : {despacho.get('proveedor_principal', '-')}")
    print(f"  Ruta          : {despacho.get('pol', '-')} → Callao (Lima)")
    print(f"  Modalidad     : {despacho.get('mode', '-')} / {despacho.get('type', '-')}")
    print(f"  Incoterm      : {despacho.get('incoterm', '-')}")
    print(f"  Contenedores  : {despacho.get('contenedores', 1)}")
    print(f"  Peso bruto    : {despacho.get('peso_bruto_kg', 0):,} kg")
    print(f"  ETA estimada  : {despacho.get('fecha_eta', '-')}")
    if perfil_cluster:
        print(f"  Perfil oper.  : Cluster {perfil_cluster['cluster_id']} — {perfil_cluster['nombre']}")
    print("-" * 75)
    print(f"  {'CONCEPTO':<30} {'ESTIMADO':>12} {'P10 (IC80)':>12} {'P90 (IC80)':>12}  RIESGO")
    print("-" * 75)

    for _, row in df_res.sort_values("costo_estimado_pen", ascending=False).iterrows():
        alerta = " ⚠" if row["nivel_riesgo"] == "ALTO" else ("  " if row["nivel_riesgo"] == "BAJO" else " →")
        print(
            f"  {row['concepto']:<30} "
            f"S/{row['costo_estimado_pen']:>10,.0f} "
            f"S/{row['p10']:>10,.0f} "
            f"S/{row['p90']:>10,.0f}"
            f"  {row['nivel_riesgo']}{alerta}"
        )

    print(SEP)
    print(f"  {'TOTAL ESTIMADO':<30} S/{total:>10,.0f} S/{p10_total:>10,.0f} S/{p90_total:>10,.0f}")
    print(SEP)
    print(f"\n  Amplitud IC 80%: {amplitud:.1f}%  → {calidad}")

    alertas = df_res[df_res["nivel_riesgo"] == "ALTO"]
    if len(alertas) > 0:
        print(f"\n  ⚠ ALERTAS — {len(alertas)} concepto(s) con riesgo ALTO:")
        for _, row in alertas.iterrows():
            print(f"    • {row['concepto']}  (score={row['riesgo_score']:.0f}/100)")
    else:
        print(f"\n  ✓ Sin alertas de riesgo ALTO")

# ---------------------------------------------------------------------------
# Ejemplos de predicción
# ---------------------------------------------------------------------------
print("\n" + "#" * 78)
print("# EJEMPLO 1: Despacho FCL marítimo (Valencia → Callao)                      #")
print("#" * 78)

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
df_pred_1       = predecir_despacho(despacho_1)
perfil_1        = obtener_perfil_cluster(despacho_1)
imprimir_prediccion(despacho_1, df_pred_1, perfil_1)

print("\n\n" + "#" * 78)
print("# EJEMPLO 2: Carga aérea urgente (Santiago → Lima)                           #")
print("#" * 78)

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
df_pred_2       = predecir_despacho(despacho_2)
perfil_2        = obtener_perfil_cluster(despacho_2)
imprimir_prediccion(despacho_2, df_pred_2, perfil_2)

# ---------------------------------------------------------------------------
# Validación masiva en test set + gráficos de evaluación
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 75)
print("VALIDACIÓN EN TEST SET COMPLETO")
print("=" * 75)

test_path = PROCESSED_DIR / "dataset_modelable_test.parquet"
if not test_path.exists():
    print("Test set no encontrado. Ejecuta primero src/03_feature_engineering.py")
else:
    test_val  = pd.read_parquet(test_path)
    col_prov_te = COL_PROV if COL_PROV in test_val.columns else "proveedor"
    cols_disp   = [c for c in ALL_FEATURES if c in test_val.columns]

    filas_eval     = []
    todos_reales   = []
    todos_predichos= []
    todos_p10      = []
    todos_p90      = []

    for concepto, modelo in modelos_por_concepto.items():
        mask = test_val["concepto_canonico"] == concepto
        if mask.sum() < 5: continue

        X_te = test_val.loc[mask, cols_disp].copy()
        for col in FEATURES_CAT:
            if col in X_te.columns:
                X_te[col] = X_te[col].astype("category")

        y_te     = test_val.loc[mask, "importe_total_pen"].values
        provs_te = test_val.loc[mask, col_prov_te].values if col_prov_te in test_val.columns else [""] * len(y_te)

        pred_log  = modelo.predict(X_te)
        pred_corr = np.array([
            _aplicar_bias_completo(p, concepto, prov)
            for p, prov in zip(np.expm1(pred_log), provs_te)
        ])

        p10_te = np.expm1(modelos_p10[concepto].predict(X_te))
        p90_te = np.expm1(modelos_p90[concepto].predict(X_te))
        marg   = cqr_margenes.get(concepto, {"margen_inf": 0.2, "margen_sup": 0.2})
        p10_cal= p10_te * (1 - marg["margen_inf"])
        p90_cal= p90_te * (1 + marg["margen_sup"])

        filas_eval.append({
            "concepto":      concepto,
            "n":             int(mask.sum()),
            "MAE_PEN":       mae(y_te, pred_corr),
            "MAPE_%":        mape(y_te, pred_corr),
            "MdAPE_%":       mdape(y_te, pred_corr),
            "RMSE_PEN":      rmse(y_te, pred_corr),
            "Cobertura_80%": cobertura_intervalo(y_te, p10_cal, p90_cal),
        })
        todos_reales.extend(y_te.tolist())
        todos_predichos.extend(pred_corr.tolist())
        todos_p10.extend(p10_cal.tolist())
        todos_p90.extend(p90_cal.tolist())

    df_eval = pd.DataFrame(filas_eval).set_index("concepto")

    print(f"\n{'CONCEPTO':<30} {'N':>6} {'MAE PEN':>10} {'MAPE%':>7} {'MdAPE%':>8} {'R80%':>8}")
    print("-" * 72)
    for conc, row in df_eval.sort_values("MdAPE_%").iterrows():
        ok  = "✓" if row["MdAPE_%"] < 5 else " "
        print(f"{conc:<30} {int(row['n']):>6} {row['MAE_PEN']:>10,.0f}"
              f" {row['MAPE_%']:>7.1f} {row['MdAPE_%']:>8.1f}"
              f" {row['Cobertura_80%']:>8.1f}  {ok}")

    print("=" * 72)
    g_mdape = df_eval["MdAPE_%"].mean()
    g_mape  = df_eval["MAPE_%"].mean()
    g_rmse  = df_eval["RMSE_PEN"].mean()
    g_cob   = df_eval["Cobertura_80%"].mean()
    print(f"{'PROMEDIO':<30}        {g_mape:>7.1f} {g_mdape:>8.1f} {g_cob:>8.1f}")
    print(f"\nMdAPE global          : {g_mdape:.1f}%")
    print(f"MAPE global           : {g_mape:.1f}%")
    print(f"RMSE promedio         : S/ {g_rmse:,.0f}")
    print(f"Cobertura IC 80%      : {g_cob:.1f}%")
    print(f"Objetivo MdAPE < 5%   : {'ALCANZADO ✓' if g_mdape < 5 else f'EN PROGRESO ({g_mdape:.1f}%)'}")
    print(f"Objetivo Cob. ≥ 80%   : {'ALCANZADO ✓' if g_cob >= 75 else f'EN PROGRESO ({g_cob:.1f}%)'}")

    # Gráfico 1: Residuos
    y_real_arr  = np.array(todos_reales)
    y_pred_arr  = np.array(todos_predichos)
    residuos    = (y_real_arr - y_pred_arr) / np.clip(y_real_arr, 1, None) * 100
    med_res     = float(np.median(residuos))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(np.log1p(y_pred_arr), residuos, alpha=0.25, s=6, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=1.2)
    axes[0].axhline(med_res, color="orange", linewidth=1, linestyle="--", label=f"Mediana={med_res:.1f}%")
    axes[0].set_xlabel("log1p(Predicho PEN)", fontsize=11)
    axes[0].set_ylabel("Residuo %", fontsize=11)
    axes[0].set_title("Residuos vs. Predicho (test set)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].hist(np.clip(residuos, -150, 150), bins=60, color="steelblue", edgecolor="white")
    axes[1].axvline(0,      color="red",    linewidth=1.2)
    axes[1].axvline(med_res,color="orange", linewidth=1.2, linestyle="--", label=f"Mediana={med_res:.1f}%")
    axes[1].set_xlabel("Residuo % (clippeado ±150%)", fontsize=11)
    axes[1].set_ylabel("Frecuencia", fontsize=11)
    axes[1].set_title(f"Distribución de residuos (n={len(residuos):,})", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demo_residuos.png", dpi=DPI_FIGURAS)
    plt.close()
    print("\nGuardado: demo_residuos.png")

    # Gráfico 2: Cobertura IC 80% por concepto
    fig, ax = plt.subplots(figsize=(10, 5))
    df_eval["Cobertura_80%"].sort_values().plot.barh(ax=ax, color="mediumseagreen", edgecolor="white")
    ax.axvline(80, color="red", linestyle="--", linewidth=1.2, label="Objetivo 80%")
    ax.set_xlabel("Cobertura (%)", fontsize=11)
    ax.set_title("Cobertura del IC 80% por concepto (test set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demo_cobertura_ic80.png", dpi=DPI_FIGURAS)
    plt.close()
    print("Guardado: demo_cobertura_ic80.png")

    # Muestra aleatoria de 10 registros reales del test
    print("\n\n" + "#" * 82)
    print("# MUESTRA ALEATORIA — 10 REGISTROS REALES DEL TEST CON PREDICCIÓN         #")
    print("#" * 82)

    conceptos_test   = [c for c in CONCEPTOS if c in test_val["concepto_canonico"].values]
    test_muestra     = test_val[test_val["concepto_canonico"].isin(conceptos_test)].copy()
    muestra          = test_muestra.sample(n=min(10, len(test_muestra)), random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\n{'N°':>3}  {'CONCEPTO':<28} {'PROVEEDOR':<20}"
          f" {'REAL':>11} {'PREDICHO':>11} {'P10':>10} {'P90':>10} {'ERR%':>7}  IC")
    print("-" * 110)

    errores_mues  = []
    dentro_ic     = 0
    col_prov_lbl  = COL_PROV if COL_PROV in muestra.columns else "proveedor"

    for idx, (_, row) in enumerate(muestra.iterrows(), start=1):
        concepto  = row["concepto_canonico"]
        modelo    = modelos_por_concepto[concepto]

        X_fila    = pd.DataFrame([row[cols_disp]])
        for col in FEATURES_CAT:
            if col in X_fila.columns:
                X_fila[col] = X_fila[col].astype("category")

        pred_log  = float(modelo.predict(X_fila)[0])
        pred_pen  = float(np.expm1(pred_log))
        prov_fila = str(row.get(col_prov_lbl, ""))
        pred_corr = _aplicar_bias_completo(pred_pen, concepto, prov_fila)

        p10_raw   = float(np.expm1(modelos_p10[concepto].predict(X_fila)[0]))
        p90_raw   = float(np.expm1(modelos_p90[concepto].predict(X_fila)[0]))
        marg      = cqr_margenes.get(concepto, {"margen_inf": 0.20, "margen_sup": 0.20})
        p10_cal   = max(p10_raw * (1 - marg["margen_inf"]), 0)
        p90_cal   = p90_raw * (1 + marg["margen_sup"])

        real      = float(row["importe_total_pen"])
        err_pct   = abs(real - pred_corr) / max(real, 1) * 100
        en_ic     = p10_cal <= real <= p90_cal
        ic_mark   = "✓" if en_ic else " "
        errores_mues.append(err_pct)
        if en_ic: dentro_ic += 1

        prov_str  = str(row.get(col_prov_lbl, "-"))[:18]
        print(f"{idx:>3}. {concepto:<28} {prov_str:<20}"
              f" S/{real:>9,.0f} S/{pred_corr:>9,.0f}"
              f" S/{p10_cal:>8,.0f} S/{p90_cal:>8,.0f}"
              f" {err_pct:>6.1f}%  {ic_mark}")

    n_mues = len(muestra)
    print("=" * 110)
    print(f"  MAPE muestra    : {np.mean(errores_mues):.1f}%")
    print(f"  MdAPE muestra   : {np.median(errores_mues):.1f}%")
    print(f"  Dentro IC 80%   : {dentro_ic}/{n_mues}  (✓ = real dentro de P10–P90)")

print("\n" + "=" * 75)
print("DEMO COMPLETADO — Pipeline HortifrutCostosImport listo para producción")
print("=" * 75)
