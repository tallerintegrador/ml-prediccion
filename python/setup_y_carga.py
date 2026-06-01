"""
setup_y_carga.py — Pipeline de carga y unificación de datos históricos de importación.

Ejecutar: python python/setup_y_carga.py
Salida  : data/processed/dataset_unificado.parquet
"""

# --- celda 1 ---
# Imports y configuración general
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# --- celda 2 ---
# Rutas del proyecto
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REF_DIR = BASE_DIR / "referencias"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for _d in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR / "figures"]:
    _d.mkdir(parents=True, exist_ok=True)

# --- celda 3 ---
# Tasas de cambio BCRP (promedio anual): USD → PEN y EUR → PEN
TC_USD_PEN = {
    2019: 3.337, 2020: 3.495, 2021: 3.881, 2022: 3.835,
    2023: 3.743, 2024: 3.780, 2025: 3.770, 2026: 3.760,
}
TC_EUR_PEN = {
    2019: 3.748, 2020: 3.950, 2021: 4.576, 2022: 3.980,
    2023: 4.100, 2024: 4.150, 2025: 4.120, 2026: 4.100,
}


def convertir_a_pen(monto, moneda, año):
    """Convierte monto a PEN usando la tasa del año correspondiente."""
    try:
        monto = float(str(monto).replace(",", ".").strip())
    except (ValueError, TypeError):
        return np.nan
    if pd.isna(monto):
        return np.nan
    moneda = str(moneda).strip().upper()
    año = int(año) if not pd.isna(año) else 2024
    if moneda == "PEN":
        return monto
    elif moneda == "USD":
        return monto * TC_USD_PEN.get(año, TC_USD_PEN[2024])
    elif moneda == "EUR":
        return monto * TC_EUR_PEN.get(año, TC_EUR_PEN[2024])
    else:
        return monto * TC_USD_PEN.get(año, TC_USD_PEN[2024])  # fallback USD

# --- celda 4 ---
# Mapeo de columnas del status (varía por año) → nombre canónico interno
STATUS_COLUMN_MAP = {
    # Identificador de operación
    "REF":           "nro_operacion",   # 2019-2020
    "OP":            "nro_operacion",   # 2021-2023
    "OPERATION":     "nro_operacion",   # 2024
    "OP REFERENCE":  "nro_operacion",   # 2025-2026
    # Campaña / temporada
    "CAMPA\xd1A":   "campaña",          # 2021-2023 (codificación)
    "CAMPAÑA":       "campaña",
    "SEASON":        "campaña",         # 2024-2026
    # Proveedor
    "Proveedor":         "proveedor_principal",  # 2019
    "SUPPLIER":          "proveedor_principal",
    # Categoría de producto
    "TIPO PROD":     "categoria",       # 2019-2020
    "CATEGORY":      "categoria",
    "CATEGORIA":     "categoria",       # 2023
    # Producto
    "PRODUCTO":      "producto",
    "PRODUCT":       "producto",
    # Contenedores
    "QTY CTNR":      "contenedores",   # 2021
    "CTNR QTY":      "contenedores",
    "TYPE CTNR":     "tipo_contenedor",
    "CTNR TYPE":     "tipo_contenedor",
    # Peso
    " PESO BRUTO ":  "peso_bruto",
    # Transportista / agente
    "AGENTE DE CARGA":   "ffw",        # 2019
    "FFW":               "ffw",
    "NAVIERA / AEROLINEA": "naviera",
    "SHIPPING LINE":     "naviera",
    # Agencia de aduana
    "AG. ADUANA":        "agencia_aduana",
    "CUSTOMS AGENT":     "agencia_aduana",
    # Fechas aduaneras
    "FECHA IE":          "fecha_ie",
    "IE DATE":           "fecha_ie",
    "FECHA LEVANTE":     "fecha_levante",
    "CUSTOMS RELEASE DATE": "fecha_levante",
    "FECHA LIBERACION SENASA": "fecha_liberacion_senasa",
    "SENASA RELEASE DATE":    "fecha_liberacion_senasa",
    # Entrega final
    "PUNTO DE LLEGADA":      "punto_llegada",
    "FINAL DELIVERY POINT":  "punto_llegada",
    # Montos
    "AMOUNT USD ":       "amount_usd",
    "AMOUNT USD":        "amount_usd",
    "USD":               "amount_usd",          # 2019
    "Valor Final USD":   "amount_usd_final",
    # Fechas de tránsito (2019 usa ETD/ETA)
    "ETD":   "fecha_atd",
    "ETA":   "fecha_ata",
    # Otros campos
    "ESTADO":    "status",       # 2019
    "Importador": "importador",  # 2019
    "IMPORTER":  "importador",   # 2025-2026
    "FACTURA":   "factura",
    "FACT":      "factura",      # 2021
    "INVOICE":   "factura",      # 2024-2026
    "COUNTRY ORIGIN": "pais_origen",
}

# Nombres simples que son iguales en todos los años
STATUS_SIMPLE_RENAMES = {
    "BUYER":         "buyer",
    "OC":            "oc",
    "STATUS":        "status",
    "INCOTERM":      "incoterm",
    "MODE":          "mode",
    "TYPE":          "type",
    "POL":           "pol",
    "POD":           "pod",
    "BULKS":         "bultos",
    "ATD":           "fecha_atd",
    "ATA":           "fecha_ata",
    "B/L- AWB":      "bl_awb",
    "DAM":           "dam",
    "BOOKING":       "booking",
    "CNTR NR":       "cntr_nr",
    "PROYECT":       "proyect",
    "QTY":           "qty",
    "UM":            "um",
    "CUR":           "cur",
    "AMOUNT":        "amount",
    "PAYMENT TERM":  "payment_term",
    "WEEK ATA":      "week_ata",
    "WEEK ATD":      "week_atd",
}

# --- celda 5 ---
def cargar_expense_report():
    """Carga y normaliza el CSV de gastos por concepto (bd_expense_report_...)."""
    path = RAW_DIR / "bd_expense_report_importaciones_201X.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Quitar espacios residuales en nombres de columnas (BOM ya lo maneja utf-8-sig)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Campaña":          "campaña",
        "Nro. Ope.":        "nro_operacion",
        "Versión Nro. Ope.":"version_ope",
        "OC":               "oc",
        "Importador/SOC":   "importador",
        "POL":              "pol",
        "Proveedor Principal": "proveedor_principal",
        "Producto":         "producto",
        "FACTURA":          "factura",
        "BL / AWB":         "bl_awb",
        "AGENCIA DE ADUANA":"agencia_aduana",
        "ACREEDOR":         "acreedor",
        "Fecha de Emisión Doc": "fecha_emision",
        "Proveedor":        "proveedor",
        "Concepto":         "concepto",
        "Moneda":           "moneda",
        "Provisión":        "provision",
        "Monto Final":      "monto_final",
        "Igv":              "igv",
        "Importe Total":    "importe_total",
        "Monto Total USD":  "monto_total_usd",
        "Clasificación":    "clasificacion",
        "Familia de Producto": "familia_producto",
        "Gerencia":         "gerencia",
        "Comprador":        "comprador",
        "Country":          "pais_origen",
        "Market":           "market",
        "OC2":              "oc2",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Limpiar BOM de campos que puedan tenerlo (ej. primer campo con encoding)
    for col in ["campaña", "nro_operacion", "concepto"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lstrip("﻿").str.strip()

    if "campaña" in df.columns:
        df["campaña"] = pd.to_numeric(df["campaña"], errors="coerce")

    # Convertir montos a numérico
    for col in ["importe_total", "monto_final", "provision", "igv", "monto_total_usd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.strip(),
                errors="coerce",
            )

    # Columna target en PEN
    df["importe_total_pen"] = df.apply(
        lambda r: convertir_a_pen(
            r.get("importe_total", np.nan),
            r.get("moneda", "PEN"),
            r.get("campaña", 2024),
        ),
        axis=1,
    )

    if "fecha_emision" in df.columns:
        df["fecha_emision"] = pd.to_datetime(df["fecha_emision"], errors="coerce", dayfirst=True)

    # Estandarizar clave de merge
    df["nro_operacion"] = df["nro_operacion"].astype(str).str.strip().str.upper()

    print(f"  Expense report: {len(df):,} filas | {df['nro_operacion'].nunique():,} operaciones | "
          f"{df['concepto'].nunique()} conceptos distintos")
    return df

# --- celda 6 ---
def _normalizar_cols_status(df):
    """Aplica el mapeo de columnas a un DataFrame de status y elimina columnas vacías."""
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in STATUS_COLUMN_MAP.items() if k in df.columns})
    df = df.rename(columns={k: v for k, v in STATUS_SIMPLE_RENAMES.items() if k in df.columns})
    # Eliminar columnas sin nombre o completamente vacías
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed") and c != ""]]
    return df


def cargar_status_reports():
    """Carga los 7 CSVs de status (2019-2026) y los concatena con esquema unificado."""
    archivos = sorted(RAW_DIR.glob("report_importaciones_*.csv"))
    frames = []

    for path in archivos:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            df = _normalizar_cols_status(df)
            df["_fuente"] = path.stem
            frames.append(df)
            print(f"  {path.name}: {len(df):,} filas")
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")

    status = pd.concat(frames, ignore_index=True, sort=False)

    # Normalizar campaña: extraer año de "2022-2023" → 2022
    status["campaña"] = (
        status["campaña"].astype(str)
        .str.extract(r"(\d{4})")[0]
    )
    status["campaña"] = pd.to_numeric(status["campaña"], errors="coerce")

    # Clave de merge
    if "nro_operacion" in status.columns:
        status["nro_operacion"] = (
            status["nro_operacion"].astype(str).str.strip().str.upper()
        )

    # Fechas de tránsito y aduana
    for col in ["fecha_atd", "fecha_ata", "fecha_ie", "fecha_levante", "fecha_liberacion_senasa"]:
        if col in status.columns:
            status[col] = pd.to_datetime(status[col], errors="coerce", dayfirst=True)

    # Numéricos
    for col in ["contenedores", "bultos", "peso_bruto", "qty", "amount_usd"]:
        if col in status.columns:
            status[col] = pd.to_numeric(
                status[col].astype(str).str.replace(",", ".").str.strip(),
                errors="coerce",
            )

    print(f"\n  Status unificado: {len(status):,} filas | "
          f"{status['nro_operacion'].nunique():,} operaciones únicas")
    return status

# --- celda 7 ---
def cargar_conceptos_canonicos():
    """
    Lee el Excel de mapeo de conceptos → 13 canónicos.
    Si falla, devuelve un dict vacío (el fallback por palabras clave se aplicará).
    """
    path = REF_DIR / "conceptos_canonicos.xlsx"
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        # Detectar columnas automáticamente
        cols_lower = {c.lower(): c for c in df.columns}
        col_orig = next(
            (v for k, v in cols_lower.items() if "concepto" in k and "canon" not in k),
            df.columns[0],
        )
        col_canon = next(
            (v for k, v in cols_lower.items() if "canon" in k),
            df.columns[1] if len(df.columns) > 1 else df.columns[0],
        )
        mapping = dict(
            zip(
                df[col_orig].astype(str).str.strip().str.upper(),
                df[col_canon].astype(str).str.strip().str.upper(),
            )
        )
        n_canon = df[col_canon].nunique()
        print(f"  {len(mapping)} entradas → {n_canon} conceptos canónicos")
        return mapping
    except Exception as e:
        print(f"  No se pudo leer {path.name}: {e}. Usando reglas de palabras clave.")
        return {}

# --- celda 8 ---
def normalizar_conceptos(df, mapping):
    """
    Agrega la columna 'concepto_canonico' usando el mapping del Excel;
    donde no haya match aplica reglas por palabras clave.
    """
    concepto_up = df["concepto"].astype(str).str.strip().str.upper()
    df["concepto_canonico"] = concepto_up.map(mapping)

    # Reglas de palabras clave para los que no tuvieron match
    reglas = [
        ("AGENCIAMIENTO",   "AGENCIAMIENTO"),
        ("AGENCIA",         "AGENCIAMIENTO"),
        ("DERECHO",         "DERECHOS_IMPUESTOS"),
        ("IMPUEST",         "DERECHOS_IMPUESTOS"),
        ("AD VALOREM",      "DERECHOS_IMPUESTOS"),
        ("IGV",             "DERECHOS_IMPUESTOS"),
        ("FLETE INT",       "FLETE_INTERNACIONAL"),
        ("FREIGHT",         "FLETE_INTERNACIONAL"),
        ("TRANSPORTE",      "TRANSPORTE_LOCAL"),
        ("HANDLING",        "HANDLING"),
        ("DESCARGA",        "HANDLING"),
        ("DEPOSITO TEMP",   "DEPOSITO_TEMPORAL"),
        ("ALMACEN",         "DEPOSITO_TEMPORAL"),
        ("DEPOSITO VACI",   "DEPOSITO_VACIOS"),
        ("GATE IN",         "DEPOSITO_VACIOS"),
        ("GATE OUT",        "DEPOSITO_VACIOS"),
        ("SEGURO",          "SEGURO"),
        ("POLIZA",          "SEGURO"),
        ("VISTO BUENO",     "VISTO_BUENO"),
        ("SOBRESTADIA",     "SOBRESTADIA"),
        ("DEMURRAGE",       "SOBRESTADIA"),
        ("AFORO",           "AFORO"),
        ("INSPECCION",      "AFORO"),
    ]

    mask_nan = df["concepto_canonico"].isna()
    for clave, canon in reglas:
        m = mask_nan & concepto_up.str.contains(clave, na=False)
        df.loc[m, "concepto_canonico"] = canon
        mask_nan = df["concepto_canonico"].isna()

    df.loc[mask_nan, "concepto_canonico"] = "OTROS_GASTOS"
    return df

# --- celda 9 ---
def merge_datasets(expense_df, status_df):
    """
    Une expense + status por nro_operacion (left join desde expense).
    Filtra registros sin match y campaña 2019 (datos incompletos).
    """
    # Deduplicar status: una fila por operación (info logística)
    status_cols_utiles = [
        c for c in [
            "nro_operacion", "campaña", "status", "importador", "buyer",
            "proveedor_principal", "categoria", "producto", "proyect",
            "qty", "um", "incoterm", "mode", "type", "pais_origen",
            "pol", "pod", "contenedores", "tipo_contenedor", "bultos",
            "ffw", "naviera", "amount_usd", "peso_bruto",
            "fecha_atd", "fecha_ata", "bl_awb", "dam",
            "agencia_aduana", "fecha_ie", "fecha_levante",
            "fecha_liberacion_senasa", "punto_llegada",
        ]
        if c in status_df.columns
    ]
    status_dedup = (
        status_df[status_cols_utiles]
        .drop_duplicates(subset="nro_operacion")
    )

    merged = expense_df.merge(
        status_dedup,
        on="nro_operacion",
        how="left",
        suffixes=("", "_st"),
    )

    # Rellenar campaña faltante desde el status
    if "campaña_st" in merged.columns:
        merged["campaña"] = merged["campaña"].fillna(merged["campaña_st"])
        merged.drop(columns=["campaña_st"], inplace=True)

    n_total = len(merged)

    # Mantener solo filas con algún dato logístico del status
    has_logistics = (
        merged.get("mode", pd.Series(dtype=object)).notna() |
        merged.get("incoterm", pd.Series(dtype=object)).notna() |
        merged.get("pol", pd.Series(dtype=object)).notna()
    )
    merged = merged[has_logistics]

    # Excluir campaña 2019
    merged = merged[merged["campaña"].fillna(0) != 2019]

    print(f"  {n_total:,} → {len(merged):,} filas tras merge y filtros")
    print(f"  Operaciones únicas: {merged['nro_operacion'].nunique():,}")
    return merged

# --- celda 10 ---
def resumen_dataset(df):
    """Imprime un resumen ejecutivo del dataset unificado."""
    print(f"\n{'='*60}")
    print(f"DATASET UNIFICADO — RESUMEN")
    print(f"{'='*60}")
    print(f"  Filas       : {len(df):,}")
    print(f"  Columnas    : {df.shape[1]}")
    print(f"  Operaciones : {df['nro_operacion'].nunique():,}")
    print(f"  Campañas    : {sorted(df['campaña'].dropna().unique().astype(int).tolist())}")

    print(f"\n  Distribución por Concepto Canónico:")
    dist = (
        df.groupby("concepto_canonico")
        .agg(
            n_filas=("importe_total_pen", "count"),
            costo_mediano_pen=("importe_total_pen", "median"),
            costo_total_kpen=("importe_total_pen", lambda x: x.sum() / 1000),
        )
        .sort_values("n_filas", ascending=False)
    )
    print(dist.round(0).to_string())

    costo_total = df["importe_total_pen"].sum()
    print(f"\n  Costo total histórico: S/ {costo_total:,.0f}")
    print(f"  Nulos en target: {df['importe_total_pen'].isna().sum():,}")

# --- celda 11 ---
def main():
    print("=" * 60)
    print("SETUP Y CARGA — HortifrutCostosImport")
    print("=" * 60)

    print("\n[1] Cargando expense report...")
    expense = cargar_expense_report()

    print("\n[2] Cargando status reports (7 archivos)...")
    status = cargar_status_reports()

    print("\n[3] Cargando mapeo de conceptos canónicos...")
    mapping_conceptos = cargar_conceptos_canonicos()

    print("\n[4] Normalizando conceptos...")
    expense = normalizar_conceptos(expense, mapping_conceptos)
    print(f"  Conceptos canónicos asignados: {expense['concepto_canonico'].nunique()} únicos")

    print("\n[5] Unificando datasets (merge expense + status)...")
    dataset = merge_datasets(expense, status)

    resumen_dataset(dataset)

    out_path = PROCESSED_DIR / "dataset_unificado.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"\nGuardado: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
