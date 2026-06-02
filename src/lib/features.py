# -*- coding: utf-8 -*-
"""
features.py — Ingeniería de features para el modelo de costos de servicios.

Decisiones (derivadas del EDA, reports/10..13):
  - log1p para variables fuertemente sesgadas (amount_usd, qty, bulks).
  - imputación: numéricas -> mediana (train); ctnr_qty -> 0 en aéreo (faltante estructural).
  - categóricas baja card. -> one-hot (con agrupación de categorías raras via min_frequency).
  - categóricas alta card. -> target encoding suavizado (sklearn TargetEncoder, cross-fit).
  - temporales: mes de arribo cíclico (sin/cos), anio (tendencia/inflación).
  - se EXCLUYE peso_bruto (90.9% nulo, ~9% disponible al predecir) y variables de leakage.

Sin leakage: ninguna columna conocida sólo tras la facturación/liquidación entra como feature.
El encoding se ajusta SIEMPRE sobre el set de entrenamiento (en CV, sólo años <= t).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, FunctionTransformer

# --------------------------------------------------------------------------- #
# Grupos de columnas
# --------------------------------------------------------------------------- #
NUM_LOG   = ["amount_usd", "qty", "bulks"]                       # sesgadas -> log1p
NUM_PLAIN = ["ctnr_qty", "payment_term", "transit_days",
             "n_oc_distintas", "n_productos"]
DERIVED   = ["anio", "mes_sin", "mes_cos"]                       # temporales derivadas
FLAGS     = ["es_aereo", "requiere_senasa", "tiene_seguro"]
CAT_OH    = ["mode", "type", "incoterm", "incoterm_grupo", "categoria_canon",
             "customs_agent", "pod"]                             # baja/media card -> one-hot
CAT_TE    = ["pol", "supplier", "shipping_line", "depot", "ffw",
             "punto_llegada", "buyer", "pais_origen"]            # alta card -> target enc

ALL_INPUT = NUM_LOG + NUM_PLAIN + DERIVED + FLAGS + CAT_OH + CAT_TE


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-procesado independiente del split (no usa el target ni estadísticos de train):
    imputación estructural de contenedores en aéreo, normalización de categóricas,
    derivación temporal cíclica. Devuelve un DataFrame con ALL_INPUT disponibles."""
    d = df.copy()

    # contenedores: en aéreo el nulo NO es faltante real -> 0
    if "ctnr_qty" in d:
        air = d["mode"].astype("string").str.upper().eq("AIR")
        d["ctnr_qty"] = d["ctnr_qty"].where(~(air & d["ctnr_qty"].isna()), 0.0)

    # temporales cíclicas (mes de arribo) + anio numérico
    mes = d["ata"].dt.month if "ata" in d else d.get("mes_arribo")
    mes = pd.to_numeric(mes, errors="coerce")
    d["mes_sin"] = np.sin(2 * np.pi * mes / 12.0)
    d["mes_cos"] = np.cos(2 * np.pi * mes / 12.0)
    d["anio"] = pd.to_numeric(d.get("anio"), errors="coerce")

    # país de origen: imputar faltante a etiqueta explícita (POL ya aporta origen)
    if "pais_origen" in d:
        d["pais_origen"] = d["pais_origen"].astype("string")

    # categóricas -> string, faltante explícito 'DESCONOCIDO'
    for c in CAT_OH + CAT_TE:
        if c in d:
            s = d[c].astype("string").str.strip().str.upper()
            d[c] = s.where(s.notna() & (s != ""), "DESCONOCIDO").astype("object")

    # flags binarias robustas
    for c in FLAGS:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(float)

    # numéricas a float (las que falten se crean como NaN -> imputadas en el CT)
    for c in NUM_LOG + NUM_PLAIN:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan

    # log1p necesita base no negativa; clip de errores de captura negativos
    for c in NUM_LOG:
        d[c] = d[c].clip(lower=0)

    return d[ALL_INPUT]


def build_column_transformer():
    """ColumnTransformer: log1p+mediana | mediana | one-hot(min_freq) | target-encoding."""
    num_log = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("imp", SimpleImputer(strategy="median")),
    ])
    num_plain = SimpleImputer(strategy="median")
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist",
                           min_frequency=10, sparse_output=False)
    target = TargetEncoder(smooth="auto", target_type="continuous", random_state=42)

    ct = ColumnTransformer(
        transformers=[
            ("numlog", num_log, NUM_LOG),
            ("num",    num_plain, NUM_PLAIN + DERIVED + FLAGS),
            ("oh",     onehot, CAT_OH),
            ("te",     target, CAT_TE),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    ct.set_output(transform="pandas")
    return ct
