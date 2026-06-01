# -*- coding: utf-8 -*-
"""
_predecir_lib.py — utilidades compartidas para PREDECIR con el champion ya entrenado.

Carga los artefactos de models/ (champion + cuantiles + margen conformal) y expone:
  - load_artifacts()                  -> dict con modelos y meta
  - construir_fila(datos: dict)       -> DataFrame de 1+ filas listo para F.prepare()
  - predecir(art, df_raw)             -> DataFrame con servicios_pred_usd / p10 / p90

El target es `target_servicios` (costo logístico SIN tributos), modelado en log1p.
Las features se preparan EXACTAMENTE igual que en el entrenamiento (src/lib/features.py),
así que cualquier columna que falte se imputa (numéricas->mediana del CT; categóricas->DESCONOCIDO).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# --- rutas a src/lib (config, features, dataset, utils) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(os.path.dirname(_HERE), "lib")
sys.path.insert(0, _LIB)

import config as C          # noqa: E402
import features as F        # noqa: E402
import dataset as D         # noqa: E402

MODELS_DIR = C.MODELS_DIR


# --------------------------------------------------------------------------- #
# back-transform (idéntico a 08_modelado.back_transform)
# --------------------------------------------------------------------------- #
def _back_transform(pred_log, log_corr):
    return np.clip(np.expm1(pred_log + log_corr), 0, None)


def _predict_member(mem, X):
    """Predice a NIVEL un miembro 'direct' del champion (ratio se descartó en producción)."""
    if mem["tkind"] != "direct":
        raise ValueError(f"miembro tkind={mem['tkind']} no soportado en predicción directa")
    return _back_transform(mem["model"].predict(X), mem["corr"])


# --------------------------------------------------------------------------- #
# Carga de artefactos
# --------------------------------------------------------------------------- #
def load_artifacts():
    champ = joblib.load(os.path.join(MODELS_DIR, "champion_servicios.pkl"))
    q10 = joblib.load(os.path.join(MODELS_DIR, "lgbm_servicios_p10.pkl"))
    q90 = joblib.load(os.path.join(MODELS_DIR, "lgbm_servicios_p90.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return dict(members=champ["members"], champion=champ["champion"],
                q10=q10, q90=q90,
                conf_margin=float(meta["conformal_margin_log"]), meta=meta)


# --------------------------------------------------------------------------- #
# Construcción de la fila de entrada
# --------------------------------------------------------------------------- #
# columnas que F.prepare() necesita que EXISTAN (si faltan, KeyError o NaN imputado)
_COLS_CAT = F.CAT_OH + F.CAT_TE          # texto -> DESCONOCIDO si falta
_COLS_NUM = F.NUM_LOG + F.NUM_PLAIN      # numéricas -> NaN -> mediana
_COLS_FLAG = F.FLAGS                     # binarias 0/1


def construir_fila(datos):
    """Convierte uno o varios dicts en un DataFrame con TODAS las columnas que el
    pipeline espera. Acepta un dict (1 fila) o lista de dicts. Deriva es_aereo,
    incoterm_grupo, categoria_canon y mes/anio igual que dataset.load()."""
    if isinstance(datos, dict):
        datos = [datos]
    df = pd.DataFrame(datos)

    # --- asegurar que existan TODAS las columnas de entrada ---
    for c in _COLS_CAT:
        if c not in df:
            df[c] = np.nan
    for c in _COLS_NUM:
        if c not in df:
            df[c] = np.nan
    for c in _COLS_FLAG:
        if c not in df:
            df[c] = np.nan

    # --- fecha de arribo / temporales ---
    # acepta 'ata' (fecha), o 'anio' + 'mes' sueltos
    if "ata" in df:
        df["ata"] = pd.to_datetime(df["ata"], errors="coerce")
    else:
        df["ata"] = pd.NaT
    if "anio" not in df or df["anio"].isna().all():
        df["anio"] = df["ata"].dt.year
    if "mes" in df:               # mes explícito -> mes_arribo (prepare usa ata o mes_arribo)
        df["mes_arribo"] = pd.to_numeric(df["mes"], errors="coerce")
    elif "mes_arribo" not in df:
        df["mes_arribo"] = df["ata"].dt.month

    # --- derivar categoría canónica si llega 'categoria' cruda ---
    if "categoria" in df and df["categoria_canon"].isna().all():
        df["categoria_canon"] = df["categoria"].map(D._categoria_canon)

    # --- incoterm_grupo desde incoterm ---
    if df["incoterm_grupo"].isna().all() and "incoterm" in df:
        inc = df["incoterm"].astype("string").str.strip().str.upper()
        df["incoterm_grupo"] = inc.map(D._INCOTERM_GRP)

    # --- pais_origen: si no llega, usar country_origin si existe ---
    if df["pais_origen"].isna().all() and "country_origin" in df:
        df["pais_origen"] = df["country_origin"]

    # --- flags ---
    if df["es_aereo"].isna().all():
        df["es_aereo"] = (df["mode"].astype("string").str.upper() == "AIR").astype(int)
    df["requiere_senasa"] = pd.to_numeric(df["requiere_senasa"], errors="coerce").fillna(0).astype(int)
    df["tiene_seguro"] = pd.to_numeric(df["tiene_seguro"], errors="coerce").fillna(0).astype(int)
    df["es_aereo"] = pd.to_numeric(df["es_aereo"], errors="coerce").fillna(0).astype(int)

    return df


# --------------------------------------------------------------------------- #
# Predicción
# --------------------------------------------------------------------------- #
def predecir(art, df_raw):
    """Devuelve un DataFrame con la predicción puntual y el intervalo P10-P90 (USD)."""
    X = F.prepare(df_raw)
    # punto = media de los miembros del champion (ens_direct = lgbm + xgb)
    yhat = np.mean([_predict_member(m, X) for m in art["members"]], axis=0)
    # intervalo conformal P10/P90 (en log -> nivel)
    lo = np.clip(np.expm1(art["q10"].predict(X) - art["conf_margin"]), 0, None)
    hi = np.maximum(np.clip(np.expm1(art["q90"].predict(X) + art["conf_margin"]), 0, None), lo)
    return pd.DataFrame({
        "servicios_pred_usd": np.round(yhat, 1),
        "p10_usd": np.round(lo, 1),
        "p90_usd": np.round(hi, 1),
    })
