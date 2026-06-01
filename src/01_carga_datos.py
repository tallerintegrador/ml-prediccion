"""
01_carga_datos.py — CRISP-DM Fase 1: Comprensión y carga de datos.

Unifica 7 status reports anuales (schemas heterogéneos) con el EXPENSE report maestro.
Normaliza conceptos a 13 categorías canónicas. Convierte monedas a PEN (BCRP).

Prerequisito: archivos CSV/XLSX en data/raw/ y referencias/
Ejecutar    : python src/01_carga_datos.py
Salida      : data/processed/dataset_unificado.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 130)

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, REF_DIR, PROCESSED_DIR, TC_USD_PEN, TC_EUR_PEN

# ---------------------------------------------------------------------------
# Tasas de cambio
# ---------------------------------------------------------------------------

def convertir_a_pen(monto, moneda, año) -> float:
    """Convierte monto a PEN usando tasa BCRP del año."""
    try:
        monto = float(str(monto).replace(",", ".").strip())
    except (ValueError, TypeError):
        return np.nan
    if pd.isna(monto):
        return np.nan
    moneda = str(moneda).strip().upper()
    año    = int(año) if not pd.isna(año) else 2024
    if moneda == "PEN":
        return monto
    elif moneda == "USD":
        return monto * TC_USD_PEN.get(año, TC_USD_PEN[2024])
    elif moneda == "EUR":
        return monto * TC_EUR_PEN.get(año, TC_EUR_PEN[2024])
    return monto * TC_USD_PEN.get(año, TC_USD_PEN[2024])  # fallback USD

# ---------------------------------------------------------------------------
# Mapeo de columnas de status reports (schema varía por año)
# ---------------------------------------------------------------------------

STATUS_COLUMN_MAP = {
    "REF":           "nro_operacion",
    "OP":            "nro_operacion",
    "OPERATION":     "nro_operacion",
    "OP REFERENCE":  "nro_operacion",
    "CAMPA\xd1A":   "campaña",
    "CAMPAÑA":       "campaña",
    "SEASON":        "campaña",
    "Proveedor":         "proveedor_principal",
    "SUPPLIER":          "proveedor_principal",
    "TIPO PROD":     "categoria",
    "CATEGORY":      "categoria",
    "CATEGORIA":     "categoria",
    "PRODUCTO":      "producto",
    "PRODUCT":       "producto",
    "QTY CTNR":      "contenedores",
    "CTNR QTY":      "contenedores",
    "TYPE CTNR":     "tipo_contenedor",
    "CTNR TYPE":     "tipo_contenedor",
    " PESO BRUTO ":  "peso_bruto",
    "AGENTE DE CARGA":   "ffw",
    "FFW":               "ffw",
    "NAVIERA / AEROLINEA": "naviera",
    "SHIPPING LINE":     "naviera",
    "AG. ADUANA":        "agencia_aduana",
    "CUSTOMS AGENT":     "agencia_aduana",
    "FECHA IE":          "fecha_ie",
    "IE DATE":           "fecha_ie",
    "FECHA LEVANTE":     "fecha_levante",
    "CUSTOMS RELEASE DATE": "fecha_levante",
    "FECHA LIBERACION SENASA": "fecha_liberacion_senasa",
    "SENASA RELEASE DATE":    "fecha_liberacion_senasa",
    "PUNTO DE LLEGADA":      "punto_llegada",
    "FINAL DELIVERY POINT":  "punto_llegada",
    "AMOUNT USD ":       "amount_usd",
    "AMOUNT USD":        "amount_usd",
    "USD":               "amount_usd",
    "Valor Final USD":   "amount_usd_final",
    "ETD":   "fecha_atd",
    "ETA":   "fecha_ata",
    "ESTADO":    "status",
    "Importador": "importador",
    "IMPORTER":  "importador",
    "FACTURA":   "factura",
    "FACT":      "factura",
    "INVOICE":   "factura",
    "COUNTRY ORIGIN": "pais_origen",
}

STATUS_SIMPLE_RENAMES = {
    "BUYER": "buyer", "OC": "oc", "STATUS": "status", "INCOTERM": "incoterm",
    "MODE": "mode", "TYPE": "type", "POL": "pol", "POD": "pod",
    "BULKS": "bultos", "ATD": "fecha_atd", "ATA": "fecha_ata",
    "B/L- AWB": "bl_awb", "DAM": "dam", "BOOKING": "booking",
    "CNTR NR": "cntr_nr", "PROYECT": "proyect", "QTY": "qty",
    "UM": "um", "CUR": "cur", "AMOUNT": "amount",
    "PAYMENT TERM": "payment_term", "WEEK ATA": "week_ata", "WEEK ATD": "week_atd",
}

# ---------------------------------------------------------------------------
# Carga del EXPENSE report
# ---------------------------------------------------------------------------

def cargar_expense_report() -> pd.DataFrame:
    """Carga y normaliza el CSV maestro de gastos por concepto."""
    path = RAW_DIR / "bd_expense_report_importaciones_201X.csv"
    df   = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Campaña": "campaña", "Nro. Ope.": "nro_operacion",
        "Versión Nro. Ope.": "version_ope", "OC": "oc",
        "Importador/SOC": "importador", "POL": "pol",
        "Proveedor Principal": "proveedor_principal", "Producto": "producto",
        "FACTURA": "factura", "BL / AWB": "bl_awb",
        "AGENCIA DE ADUANA": "agencia_aduana", "ACREEDOR": "acreedor",
        "Fecha de Emisión Doc": "fecha_emision", "Proveedor": "proveedor",
        "Concepto": "concepto", "Moneda": "moneda",
        "Provisión": "provision", "Monto Final": "monto_final",
        "Igv": "igv", "Importe Total": "importe_total",
        "Monto Total USD": "monto_total_usd", "Clasificación": "clasificacion",
        "Familia de Producto": "familia_producto", "Gerencia": "gerencia",
        "Comprador": "comprador", "Country": "pais_origen",
        "Market": "market", "OC2": "oc2",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["campaña", "nro_operacion", "concepto"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lstrip("﻿").str.strip()

    if "campaña" in df.columns:
        df["campaña"] = pd.to_numeric(df["campaña"], errors="coerce")

    for col in ["importe_total", "monto_final", "provision", "igv", "monto_total_usd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.strip(), errors="coerce"
            )

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

    df["nro_operacion"] = df["nro_operacion"].astype(str).str.strip().str.upper()

    print(f"  Expense report: {len(df):,} filas | {df['nro_operacion'].nunique():,} operaciones | "
          f"{df['concepto'].nunique()} conceptos")
    return df

# ---------------------------------------------------------------------------
# Carga y unificación de status reports
# ---------------------------------------------------------------------------

def _normalizar_cols_status(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica mapeo de columnas y elimina columnas vacías o sin nombre."""
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in STATUS_COLUMN_MAP.items() if k in df.columns})
    df = df.rename(columns={k: v for k, v in STATUS_SIMPLE_RENAMES.items() if k in df.columns})
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed") and c != ""]]
    return df


def cargar_status_reports() -> pd.DataFrame:
    """Carga los 7 CSVs de status (2019-2026) y los concatena con schema unificado."""
    archivos = sorted(RAW_DIR.glob("report_importaciones_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron status reports en {RAW_DIR}")

    frames = []
    for path in archivos:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            df = df.dropna(how="all")           # padding de Excel
            df = _normalizar_cols_status(df)
            df["_fuente"] = path.stem
            frames.append(df)
            print(f"  {path.name}: {len(df):,} filas cargadas")
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")

    status = pd.concat(frames, ignore_index=True, sort=False)

    status["campaña"] = (
        status["campaña"].astype(str)
        .str.extract(r"(\d{4})")[0]
    )
    status["campaña"] = pd.to_numeric(status["campaña"], errors="coerce")

    if "nro_operacion" in status.columns:
        status["nro_operacion"] = status["nro_operacion"].astype(str).str.strip().str.upper()

    for col in ["fecha_atd", "fecha_ata", "fecha_ie", "fecha_levante", "fecha_liberacion_senasa"]:
        if col in status.columns:
            status[col] = pd.to_datetime(status[col], errors="coerce", dayfirst=True)

    for col in ["contenedores", "bultos", "peso_bruto", "qty", "amount_usd"]:
        if col in status.columns:
            status[col] = pd.to_numeric(
                status[col].astype(str).str.replace(",", ".").str.strip(), errors="coerce"
            )

    print(f"\n  Status unificado: {len(status):,} filas | "
          f"{status['nro_operacion'].nunique():,} operaciones únicas")
    return status

# ---------------------------------------------------------------------------
# Mapeo de conceptos canónicos
# ---------------------------------------------------------------------------

def cargar_conceptos_canonicos() -> dict:
    """Lee el Excel de mapeo concepto → canónico. Devuelve dict vacío si falla."""
    path = REF_DIR / "conceptos_canonicos.xlsx"
    try:
        df   = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        cols_lower = {c.lower(): c for c in df.columns}
        col_orig  = next((v for k, v in cols_lower.items() if "concepto" in k and "canon" not in k), df.columns[0])
        col_canon = next((v for k, v in cols_lower.items() if "canon" in k), df.columns[1] if len(df.columns) > 1 else df.columns[0])
        mapping   = dict(zip(
            df[col_orig].astype(str).str.strip().str.upper(),
            df[col_canon].astype(str).str.strip().str.upper(),
        ))
        print(f"  {len(mapping)} entradas → {df[col_canon].nunique()} conceptos canónicos")
        return mapping
    except Exception as e:
        print(f"  No se pudo leer conceptos_canonicos.xlsx: {e}. Usando reglas de palabras clave.")
        return {}


REGLAS_CANONICO = [
    ("AGENCIAMIENTO", "AGENCIAMIENTO"), ("AGENCIA", "AGENCIAMIENTO"),
    ("DERECHO",       "DERECHOS_IMPUESTOS"), ("IMPUEST", "DERECHOS_IMPUESTOS"),
    ("AD VALOREM",    "DERECHOS_IMPUESTOS"), ("IGV", "DERECHOS_IMPUESTOS"),
    ("FLETE INT",     "FLETE_INTERNACIONAL"), ("FREIGHT", "FLETE_INTERNACIONAL"),
    ("TRANSPORTE",    "TRANSPORTE_LOCAL"),
    ("HANDLING",      "HANDLING"), ("DESCARGA", "HANDLING"),
    ("DEPOSITO TEMP", "DEPOSITO_TEMPORAL"), ("ALMACEN", "DEPOSITO_TEMPORAL"),
    ("DEPOSITO VACI", "DEPOSITO_VACIOS"), ("GATE IN", "DEPOSITO_VACIOS"),
    ("GATE OUT",      "DEPOSITO_VACIOS"),
    ("SEGURO",        "SEGURO"), ("POLIZA", "SEGURO"),
    ("VISTO BUENO",   "VISTO_BUENO"),
    ("SOBRESTADIA",   "SOBRESTADIA"), ("DEMURRAGE", "SOBRESTADIA"),
    ("AFORO",         "AFORO"), ("INSPECCION", "AFORO"),
]


def normalizar_conceptos(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Agrega 'concepto_canonico' usando Excel mapping + reglas de palabras clave."""
    concepto_up = df["concepto"].astype(str).str.strip().str.upper()
    df["concepto_canonico"] = concepto_up.map(mapping)
    mask_nan = df["concepto_canonico"].isna()
    for clave, canon in REGLAS_CANONICO:
        m = mask_nan & concepto_up.str.contains(clave, na=False)
        df.loc[m, "concepto_canonico"] = canon
        mask_nan = df["concepto_canonico"].isna()
    df.loc[mask_nan, "concepto_canonico"] = "OTROS_GASTOS"
    return df

# ---------------------------------------------------------------------------
# Merge expense + status
# ---------------------------------------------------------------------------

def merge_datasets(expense_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
    """Une expense + status por nro_operacion. Filtra sin match y campaña 2019."""
    status_cols = [c for c in [
        "nro_operacion", "campaña", "status", "importador", "buyer",
        "proveedor_principal", "categoria", "producto", "proyect",
        "qty", "um", "incoterm", "mode", "type", "pais_origen",
        "pol", "pod", "contenedores", "tipo_contenedor", "bultos",
        "ffw", "naviera", "amount_usd", "peso_bruto",
        "fecha_atd", "fecha_ata", "bl_awb", "dam",
        "agencia_aduana", "fecha_ie", "fecha_levante",
        "fecha_liberacion_senasa", "punto_llegada",
    ] if c in status_df.columns]

    status_dedup = status_df[status_cols].drop_duplicates(subset="nro_operacion")

    merged = expense_df.merge(status_dedup, on="nro_operacion", how="left", suffixes=("", "_st"))

    if "campaña_st" in merged.columns:
        merged["campaña"] = merged["campaña"].fillna(merged["campaña_st"])
        merged.drop(columns=["campaña_st"], inplace=True)

    n_total = len(merged)
    has_logistics = (
        merged.get("mode", pd.Series(dtype=object)).notna() |
        merged.get("incoterm", pd.Series(dtype=object)).notna() |
        merged.get("pol", pd.Series(dtype=object)).notna()
    )
    merged = merged[has_logistics]
    merged = merged[merged["campaña"].fillna(0) != 2019]

    print(f"  Merge: {n_total:,} → {len(merged):,} filas  |  "
          f"{merged['nro_operacion'].nunique():,} operaciones")
    return merged

# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------

def resumen_dataset(df: pd.DataFrame):
    sep = "=" * 65
    print(f"\n{sep}\nDATASET UNIFICADO — RESUMEN\n{sep}")
    print(f"  Filas       : {len(df):,}")
    print(f"  Columnas    : {df.shape[1]}")
    print(f"  Operaciones : {df['nro_operacion'].nunique():,}")
    campañas = sorted(df["campaña"].dropna().unique().astype(int).tolist())
    print(f"  Campañas    : {campañas}")

    print(f"\n  {'CONCEPTO CANÓNICO':<30} {'N':>7} {'MEDIANA PEN':>14} {'TOTAL KPEN':>12}")
    print("  " + "-" * 65)
    dist = (
        df.groupby("concepto_canonico")
        .agg(n=("importe_total_pen", "count"),
             mediana=("importe_total_pen", "median"),
             total=("importe_total_pen", lambda x: x.sum() / 1_000))
        .sort_values("n", ascending=False)
    )
    for conc, row in dist.iterrows():
        print(f"  {conc:<30} {int(row['n']):>7,} {row['mediana']:>14,.0f} {row['total']:>12,.0f}")

    total_hist = df["importe_total_pen"].sum()
    print(f"\n  Costo histórico total : S/ {total_hist:,.0f}")
    print(f"  Nulos en target       : {df['importe_total_pen'].isna().sum():,}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(f"{sep}\n01_CARGA_DATOS — HortifrutCostosImport (CRISP-DM Fase 1)\n{sep}")

    print("\n[1/5] Cargando expense report...")
    expense = cargar_expense_report()

    print("\n[2/5] Cargando status reports (7 archivos anuales)...")
    status = cargar_status_reports()

    print("\n[3/5] Cargando mapeo de conceptos canónicos...")
    mapping_conceptos = cargar_conceptos_canonicos()

    print("\n[4/5] Normalizando conceptos...")
    expense = normalizar_conceptos(expense, mapping_conceptos)
    n_canonicos = expense["concepto_canonico"].nunique()
    print(f"  Conceptos canónicos asignados: {n_canonicos}")
    print(f"  Distribución:\n{expense['concepto_canonico'].value_counts().head(15).to_string()}")

    print("\n[5/5] Merge expense + status...")
    dataset = merge_datasets(expense, status)

    resumen_dataset(dataset)

    out_path = PROCESSED_DIR / "dataset_unificado.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"\n  Guardado: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
