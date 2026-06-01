# -*- coding: utf-8 -*-
"""
utils.py — Helpers de carga y limpieza para el EDA v2.

Incluye:
  - normalización de encabezados y de texto (sin acentos)
  - lectura robusta de CSV (encoding/sep por archivo, tokens de nulo)
  - parser de montos (símbolos, separadores de miles, coma decimal, multimoneda)
  - parser de fechas (varios formatos dd/mm/aaaa)
  - canonicalización de conceptos de costo
  - utilidades de reporte en markdown
"""
import re
import warnings
import unicodedata
import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- #
# Normalización de strings
# --------------------------------------------------------------------------- #
def strip_accents(s: str) -> str:
    if s is None:
        return s
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))


def norm_header(s: str) -> str:
    """Normaliza un encabezado de columna para matching: MAYÚSCULAS, sin acentos,
    sin puntos, espacios colapsados. 'Nro. Ope.' -> 'NRO OPE'; ' PESO BRUTO ' -> 'PESO BRUTO'."""
    if s is None:
        return ""
    s = strip_accents(str(s)).upper()
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_text(s) -> str:
    """Normaliza texto de celda categórica/concepto: minúsculas, sin acentos,
    espacios colapsados. Devuelve '' para NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = strip_accents(str(s)).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Mapa reverso normalizado -> canónico (construido una vez)
_REVERSE_COL_MAP = {}
for _canon, _variants in C.CANON_COLUMNS.items():
    for _v in _variants:
        _REVERSE_COL_MAP[norm_header(_v)] = _canon


def canon_column(raw_name: str):
    """Devuelve el nombre canónico de una columna cruda, o None si es huérfana."""
    return _REVERSE_COL_MAP.get(norm_header(raw_name))


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
_MONEY_CUR = {"$": "USD", "us$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
              "s/": "PEN", "s/.": "PEN", "pen": "PEN", "soles": "PEN"}


def parse_money(x, default_cur=None):
    """Parsea un monto que puede venir como texto con símbolos/separadores.
    Devuelve (valor_float, moneda_detectada|None). Soporta formato es/us
    ('221,505.80', '4.129,65', '$ 221,505.80', '€ 0.00')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return (np.nan, None)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "-", "none"):
        return (np.nan, None)
    cur = None
    low = s.lower()
    for sym, c in _MONEY_CUR.items():
        if sym in low:
            cur = c
            break
    # quitar todo lo que no sea dígito, separador o signo
    s = re.sub(r"[^\d,.\-]", "", s)
    if s in ("", "-", ".", ","):
        return (np.nan, cur or default_cur)
    # decidir separador decimal: el último de , o . que aparezca
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")
    if last_comma > last_dot:
        # coma decimal -> quitar puntos (miles), coma -> punto
        s = s.replace(".", "").replace(",", ".")
    else:
        # punto decimal -> quitar comas (miles)
        s = s.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return (np.nan, cur or default_cur)
    return (val, cur or default_cur)


def to_number(x):
    """Parsea sólo el valor numérico (ignora moneda)."""
    return parse_money(x)[0]


def parse_date(x):
    """Parsea fecha en formatos heterogéneos -> pd.Timestamp o NaT."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "-", "none", "pend", "pendiente"):
        return pd.NaT
    # intentar día primero (dd/mm/aaaa típico en los reportes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dayfirst in (True, False):
            dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
            if pd.notna(dt):
                # descartar fechas imposibles
                if 2015 <= dt.year <= 2027:
                    return dt
    return pd.NaT


def amount_to_usd(value, cur):
    """Convierte un monto a USD usando FX de respaldo (solo mercadería operativa)."""
    if pd.isna(value):
        return np.nan
    rate = C.FX_TO_USD.get(str(cur).upper() if cur else "USD", np.nan)
    return value * rate


# --------------------------------------------------------------------------- #
# Conceptos
# --------------------------------------------------------------------------- #
_COMPILED_CONCEPT_RULES = [(re.compile(p), name) for p, name in C.CONCEPT_RULES]


def canon_concept(raw_concept) -> str:
    """Canonicaliza un concepto de costo crudo aplicando CONCEPT_RULES en orden."""
    t = norm_text(raw_concept)
    if t == "":
        return C.CONCEPTO_CANON_DEFAULT
    for rgx, name in _COMPILED_CONCEPT_RULES:
        if rgx.search(t):
            return name
    return C.CONCEPTO_CANON_DEFAULT


# --------------------------------------------------------------------------- #
# Llave de operación (descubierta en el probe: YY-NNN robusto a deriva de código)
# --------------------------------------------------------------------------- #
def op_key(x):
    """Llave de unión de operación: 'YY-NNN' (año + secuencia)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    m = re.match(r"\s*(\d{2})\D*(\d{3})", str(x).upper().strip())
    return f"{m.group(1)}-{m.group(2)}" if m else None


def op_id_full(x):
    """Identificador normalizado completo (YYNNNCODE) — para detectar colisiones."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return re.sub(r"[^0-9A-Z]", "", str(x).upper().strip()) or None


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_raw(path, sep, enc):
    """Lee un CSV crudo como texto, con tokens de nulo, sin coerción de tipos."""
    return pd.read_csv(path, sep=sep, encoding=enc, dtype=str,
                       keep_default_na=False, na_values=C.NULL_TOKENS,
                       engine="python", on_bad_lines="warn")


def drop_empty_cols(df, thresh=1.0):
    """Elimina columnas totalmente vacías (>= thresh fracción de nulos)."""
    keep = [c for c in df.columns if df[c].notna().mean() < thresh or df[c].notna().any()]
    return df[keep]


# --------------------------------------------------------------------------- #
# Reporte markdown
# --------------------------------------------------------------------------- #
class MdReport:
    def __init__(self, title):
        self.lines = [f"# {title}\n"]

    def h(self, txt, level=2):
        self.lines.append(f"\n{'#' * level} {txt}\n")

    def p(self, txt=""):
        self.lines.append(txt)

    def table(self, df, floatfmt="{:.2f}", index=False, max_rows=None):
        d = df.copy()
        if max_rows:
            d = d.head(max_rows)
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        self.lines.append(d.to_markdown(index=index))

    def save(self, path):
        import io
        with io.open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in self.lines))
        return path
