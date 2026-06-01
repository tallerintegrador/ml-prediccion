"""
utils.py — Funciones y constantes compartidas entre los scripts del pipeline ML.

Importar en cualquier script con:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import limpiar_texto, mediana_porcentaje_error_absoluto, rmse, ...
"""

import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de dominio
# ---------------------------------------------------------------------------

INCOTERM_GRUPO = {"E": "GRUPO_E", "F": "GRUPO_F", "C": "GRUPO_C", "D": "GRUPO_D"}

RUTA_ORIGEN = {
    "VALENCIA": "EUROPA", "BARCELONA": "EUROPA", "ROTTERDAM": "EUROPA",
    "HAMBURG": "EUROPA", "FELIXSTOWE": "EUROPA", "GENOVA": "EUROPA",
    "VIGO": "EUROPA", "RIGA": "EUROPA", "TALLIN": "EUROPA",
    "LEIXOES": "EUROPA", "LISBON": "EUROPA", "AMSTERDAM": "EUROPA",
    "ANTWERP": "EUROPA", "BILBAO": "EUROPA", "MADRID": "EUROPA",
    "VALPARAISO": "LATAM", "SAN ANTONIO": "LATAM", "BUENOS AIRES": "LATAM",
    "CALLAO": "LATAM", "GUAYAQUIL": "LATAM", "MANZANILLO": "LATAM",
    "VERACRUZ": "LATAM", "BUENOS": "LATAM", "SANTIAGO": "LATAM",
    "LIMA": "LATAM", "MEXICO": "LATAM",
    "MIAMI": "NORTEAM", "LOS ANGELES": "NORTEAM", "NEW YORK": "NORTEAM",
    "HOUSTON": "NORTEAM", "SEATTLE": "NORTEAM",
    "SHANGHAI": "ASIA", "HONG KONG": "ASIA", "SINGAPORE": "ASIA",
    "BUSAN": "ASIA", "GUANGZHOU": "ASIA", "NINGBO": "ASIA",
}

DIAS_TRANSITO = {"AIR": 5, "LCL": 22, "FCL": 28, "COURIER": 3}

# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------

def limpiar_texto(s) -> str:
    """Normaliza un string: upper, sin acentos, sin espacios dobles."""
    if pd.isna(s):
        return "DESCONOCIDO"
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())

# ---------------------------------------------------------------------------
# Mappings categóricos
# ---------------------------------------------------------------------------

def mapear_incoterm_grupo(s: str) -> str:
    s = limpiar_texto(s)
    return INCOTERM_GRUPO.get(s[0] if s else "F", "GRUPO_F")


def mapear_ruta_origen(pol: str) -> str:
    pol = limpiar_texto(pol)
    for clave, region in RUTA_ORIGEN.items():
        if clave in pol:
            return region
    return "OTROS"


def estimar_dias_transito(mode: str) -> int:
    m = limpiar_texto(str(mode))
    for k, v in DIAS_TRANSITO.items():
        if k in m:
            return v
    return 25

# ---------------------------------------------------------------------------
# Métricas de regresión
# ---------------------------------------------------------------------------

def mediana_porcentaje_error_absoluto(y_true, y_pred) -> float:
    """
    MdAPE: mediana del porcentaje de error absoluto.
    Denominador clippeado a 1e-6 para evitar división por cero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pct = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))
    return float(np.median(pct) * 100)


def rmse(y_true, y_pred) -> float:
    """RMSE: raíz del error cuadrático medio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
