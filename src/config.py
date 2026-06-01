"""
config.py — Configuración centralizada del proyecto HortifrutCostosImport.

Importar en cualquier script con:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import BASE_DIR, MODELS_DIR, RANDOM_STATE, ...
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
REF_DIR       = BASE_DIR / "referencias"
SRC_DIR       = Path(__file__).parent

for _dir in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibilidad y configuración del modelo
# ---------------------------------------------------------------------------
RANDOM_STATE  = 42
FECHA_SPLIT   = "2025-01-01"
HALFLIFE_DIAS = 365

# ---------------------------------------------------------------------------
# Tasas de cambio BCRP (promedio anual): USD → PEN y EUR → PEN
# ---------------------------------------------------------------------------
TC_USD_PEN = {
    2019: 3.337, 2020: 3.495, 2021: 3.881, 2022: 3.835,
    2023: 3.743, 2024: 3.780, 2025: 3.770, 2026: 3.760,
}
TC_EUR_PEN = {
    2019: 3.748, 2020: 3.950, 2021: 4.576, 2022: 3.980,
    2023: 4.100, 2024: 4.150, 2025: 4.120, 2026: 4.100,
}

# ---------------------------------------------------------------------------
# Dominio: agrupaciones logísticas
# ---------------------------------------------------------------------------
INCOTERM_GRUPO = {"E": "GRUPO_E", "F": "GRUPO_F", "C": "GRUPO_C", "D": "GRUPO_D"}

RUTA_ORIGEN = {
    "VALENCIA": "EUROPA",  "BARCELONA": "EUROPA", "ROTTERDAM": "EUROPA",
    "HAMBURG":  "EUROPA",  "FELIXSTOWE": "EUROPA", "GENOVA": "EUROPA",
    "VIGO":     "EUROPA",  "RIGA": "EUROPA",       "TALLIN": "EUROPA",
    "LEIXOES":  "EUROPA",  "LISBON": "EUROPA",     "AMSTERDAM": "EUROPA",
    "ANTWERP":  "EUROPA",  "BILBAO": "EUROPA",     "MADRID": "EUROPA",
    "VALPARAISO": "LATAM", "SAN ANTONIO": "LATAM", "BUENOS AIRES": "LATAM",
    "CALLAO":   "LATAM",   "GUAYAQUIL": "LATAM",   "MANZANILLO": "LATAM",
    "VERACRUZ": "LATAM",   "SANTIAGO": "LATAM",    "LIMA": "LATAM",
    "MIAMI":    "NORTEAM", "LOS ANGELES": "NORTEAM", "NEW YORK": "NORTEAM",
    "HOUSTON":  "NORTEAM", "SEATTLE": "NORTEAM",
    "SHANGHAI": "ASIA",    "HONG KONG": "ASIA",    "SINGAPORE": "ASIA",
    "BUSAN":    "ASIA",    "GUANGZHOU": "ASIA",    "NINGBO": "ASIA",
}

DIAS_TRANSITO = {"AIR": 5, "LCL": 22, "FCL": 28, "COURIER": 3}

# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
DPI_FIGURAS    = 150
COLOR_PRIMARIO = "steelblue"
COLOR_ALERT    = "coral"
COLOR_OK       = "mediumseagreen"
PALETTE        = "Set2"
