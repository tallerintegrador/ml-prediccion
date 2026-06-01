"""
utils.py — Funciones y constantes compartidas del pipeline HortifrutCostosImport.

Importar con:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import limpiar_texto, mdape, rmse, mae, ...
"""

import unicodedata

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import INCOTERM_GRUPO, RUTA_ORIGEN, DIAS_TRANSITO

# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def limpiar_texto(s) -> str:
    """Normaliza string: upper, sin acentos, sin espacios dobles."""
    if pd.isna(s):
        return "DESCONOCIDO"
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())

# ---------------------------------------------------------------------------
# Mappings de dominio
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

def mdape(y_true, y_pred) -> float:
    """MdAPE: mediana del porcentaje de error absoluto (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pct    = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))
    return float(np.median(pct) * 100)

mediana_porcentaje_error_absoluto = mdape  # alias de compatibilidad


def mape(y_true, y_pred) -> float:
    """MAPE: media del porcentaje de error absoluto (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pct    = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))
    return float(np.mean(pct) * 100)


def rmse(y_true, y_pred) -> float:
    """RMSE: raíz del error cuadrático medio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """MAE: error absoluto medio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# ---------------------------------------------------------------------------
# Métricas de clasificación auxiliares
# ---------------------------------------------------------------------------

def cobertura_intervalo(y_true, p_inf, p_sup) -> float:
    """Proporción de valores reales dentro del intervalo [p_inf, p_sup]."""
    y_true = np.asarray(y_true, dtype=float)
    p_inf  = np.asarray(p_inf,  dtype=float)
    p_sup  = np.asarray(p_sup,  dtype=float)
    return float(((y_true >= p_inf) & (y_true <= p_sup)).mean() * 100)

# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------

def calcular_psi_numerico(x_base, x_prod, n_bins: int = 10) -> float:
    """PSI para una feature continua, usando deciles del conjunto base."""
    x_base = pd.Series(x_base).dropna()
    x_prod = pd.Series(x_prod).dropna()
    if len(x_base) < 10 or len(x_prod) < 5:
        return float("nan")
    bins      = np.percentile(x_base, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    bins = np.unique(bins)
    if len(bins) < 3:
        return float("nan")
    p_base = np.clip(np.histogram(x_base, bins=bins)[0] / len(x_base), 1e-6, None)
    p_prod = np.clip(np.histogram(x_prod, bins=bins)[0] / len(x_prod), 1e-6, None)
    return float(np.sum((p_prod - p_base) * np.log(p_prod / p_base)))


def calcular_psi_categorico(x_base, x_prod) -> float:
    """PSI para una feature categórica."""
    x_base = pd.Series(x_base).dropna().astype(str)
    x_prod = pd.Series(x_prod).dropna().astype(str)
    cats   = set(x_base) | set(x_prod)
    psi    = 0.0
    for c in cats:
        pb  = max((x_base == c).mean(), 1e-6)
        pp  = max((x_prod == c).mean(), 1e-6)
        psi += (pp - pb) * np.log(pp / pb)
    return float(psi)


def interpretar_psi(psi) -> str:
    if pd.isna(psi):
        return "SIN DATOS"
    elif psi < 0.10:
        return "ESTABLE"
    elif psi < 0.25:
        return "CAMBIO MODERADO"
    else:
        return "DRIFT CRITICO"
