# -*- coding: utf-8 -*-
"""
01_carga.py — Carga, armonización y consolidación (secciones 4.0.3 y 4.1).

  - Lee los 7 reportes operativos (deriva de esquema por año) y los armoniza a
    un esquema canónico común; resuelve colisiones de columnas dentro de un archivo.
  - Lee el expense report y normaliza montos/conceptos.
  - Recodifica nulos, parsea montos (multimoneda) y fechas.
  - Consolida los operativos en una sola tabla a nivel de LÍNEA (1 fila por OC/producto).
  - Genera:
       data_csv/processed/operativo_lineas.parquet
       data_csv/processed/expense_lineas.parquet
       reports/02_armonizacion_columnas.md
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils as U

NUMERIC_COLS = ["qty", "amount", "amount_usd", "ctnr_qty", "bulks",
                "peso_bruto", "payment_term"]
DATE_COLS = ["cargo_ready", "quote_request_date", "assignment_date", "pick_up_date",
             "atd", "ata", "ie_date", "fecha_num_dam", "fecha_descarga", "fecha_tarja",
             "fecha_levante", "senasa_inspeccion", "senasa_liberacion", "demurrage_exp",
             "fecha_retiro_t1", "fecha_entrega_base_t2", "fecha_fin_viaje",
             "receipt_confirmation", "fecha_liquidacion", "fecha_registro_gastos"]


def harmonize_file(df, schema):
    """Renombra columnas crudas a canónicas resolviendo colisiones por prioridad
    (orden en CANON_COLUMNS). Devuelve (df_canon, mapping_dict, orphans_list)."""
    # candidatas: raw -> (canon, prioridad)
    cand = {}
    for raw in df.columns:
        canon = U.canon_column(raw)
        if canon is None:
            continue
        variants = [U.norm_header(v) for v in C.CANON_COLUMNS[canon]]
        prio = variants.index(U.norm_header(raw)) if U.norm_header(raw) in variants else 99
        cand[raw] = (canon, prio)
    # resolver colisiones: por cada canon, gana la de menor prioridad
    chosen = {}
    for raw, (canon, prio) in cand.items():
        if canon not in chosen or prio < chosen[canon][1]:
            chosen[canon] = (raw, prio)
    rename = {raw: canon for canon, (raw, _) in chosen.items()}
    mapping = {canon: raw for canon, (raw, _) in chosen.items()}
    orphans = [c for c in df.columns if c not in rename]
    out = df.rename(columns=rename)
    # conservar solo columnas canónicas presentes
    canon_present = [c for c in C.CANON_COLUMNS if c in out.columns]
    out = out[canon_present].copy()
    out["schema"] = schema
    return out, mapping, orphans


def main():
    frames = []
    mapping_matrix = {}   # canon -> {schema -> raw}
    orphans_by_file = {}

    # -------- operativos --------
    for fname, cfg in C.OPERATIVE_FILES.items():
        path = os.path.join(C.RAW_DIR, fname)
        raw = U.read_raw(path, cfg["sep"], cfg["enc"])
        df, mapping, orphans = harmonize_file(raw, cfg["schema"])
        df["op_key"] = raw[cfg["id_col"]].map(U.op_key)
        df["op_id_full"] = raw[cfg["id_col"]].map(U.op_id_full)
        df["source_file"] = fname
        frames.append(df)
        for canon, rawname in mapping.items():
            mapping_matrix.setdefault(canon, {})[cfg["schema"]] = rawname
        orphans_by_file[cfg["schema"]] = orphans
        print(f"{cfg['schema']:>10}: {raw.shape[0]:>4} filas | "
              f"{len(mapping)} canónicas | {len(orphans)} huérfanas")

    op = pd.concat(frames, ignore_index=True, sort=False)
    # asegurar todas las columnas canónicas
    for c in C.CANON_COLUMNS:
        if c not in op.columns:
            op[c] = np.nan

    # parseo de tipos
    for c in NUMERIC_COLS:
        if c in op.columns:
            op[c] = op[c].map(U.to_number)
    for c in DATE_COLS:
        if c in op.columns:
            op[c] = op[c].map(U.parse_date)

    # derivados básicos
    op["anio_op"] = op["op_key"].str.slice(0, 2).map(
        lambda y: 2000 + int(y) if pd.notna(y) and str(y).isdigit() else np.nan)
    # amount_usd de respaldo donde falte (p. ej. 2021): convertir amount con moneda
    mask = op["amount_usd"].isna() & op["amount"].notna()
    op.loc[mask, "amount_usd"] = [
        U.amount_to_usd(a, m) for a, m in zip(op.loc[mask, "amount"], op.loc[mask, "moneda"])]

    op.to_parquet(os.path.join(C.PROC_DIR, "operativo_lineas.parquet"), index=False)
    print(f"\noperativo_lineas: {op.shape}")

    # -------- expense --------
    epath = os.path.join(C.RAW_DIR, C.EXPENSE_FILE)
    exp = U.read_raw(epath, C.EXPENSE_CFG["sep"], C.EXPENSE_CFG["enc"])
    exp = exp.rename(columns={
        "Nro. Ope.": "nro_ope", "Campaña": "campania_exp", "OC": "oc",
        "Concepto": "concepto_raw", "Moneda": "moneda", "Monto Final": "monto_final",
        "Igv": "igv", "Importe Total": "importe_total", "Monto Total USD": "monto_usd",
        "Proveedor": "proveedor_costo", "AGENCIA DE ADUANA": "agencia_aduana",
        "Tipo de Doc.": "tipo_doc", "Fecha de Envío Liquidación": "fecha_liq",
        "Producto": "producto_exp"})
    exp["op_key"] = exp["nro_ope"].map(U.op_key)
    exp["op_id_full"] = exp["nro_ope"].map(U.op_id_full)
    exp["concepto_canon"] = exp["concepto_raw"].map(U.canon_concept)
    for c in ["monto_final", "igv", "importe_total", "monto_usd"]:
        exp[c] = exp[c].map(U.to_number)
    keep = ["op_key", "op_id_full", "nro_ope", "campania_exp", "oc", "proveedor_costo",
            "concepto_raw", "concepto_canon", "moneda", "monto_final", "igv",
            "importe_total", "monto_usd", "tipo_doc", "agencia_aduana", "producto_exp"]
    exp = exp[keep]
    exp.to_parquet(os.path.join(C.PROC_DIR, "expense_lineas.parquet"), index=False)
    print(f"expense_lineas: {exp.shape}")

    # -------- reporte de armonización --------
    R = U.MdReport("4.0.3 / 4.1 — Armonización de columnas (deriva de esquema)")
    R.p("Mapa canónico reconstruido **desde cero** inspeccionando los 7 esquemas "
        "operativos. Cada fila es una variable canónica; cada columna, el nombre "
        "crudo real en ese archivo (— = ausente).")
    schemas = [cfg["schema"] for cfg in C.OPERATIVE_FILES.values()]
    rows = []
    for canon in C.CANON_COLUMNS:
        if canon not in mapping_matrix:
            continue
        row = {"canónica": canon}
        for s in schemas:
            row[s] = mapping_matrix[canon].get(s, "—")
        rows.append(row)
    mat = pd.DataFrame(rows)
    R.h("Matriz de armonización canónica × esquema")
    R.p(mat.to_markdown(index=False))

    R.h("Resumen de cobertura")
    cov = pd.DataFrame({
        "esquema": schemas,
        "columnas_canónicas": [sum(1 for c in mapping_matrix if s in mapping_matrix[c]) for s in schemas],
        "columnas_huérfanas": [len(orphans_by_file[s]) for s in schemas],
    })
    R.p(cov.to_markdown(index=False))

    R.h("Columnas huérfanas por esquema (no mapeadas a canónica)")
    for s in schemas:
        orphs = [o for o in orphans_by_file[s] if not o.lower().startswith("unnamed")]
        R.p(f"\n**{s}** ({len(orphs)} útiles, excluye 'Unnamed'): " +
            (", ".join(f"`{o}`" for o in orphs) if orphs else "—"))

    out = R.save(os.path.join(C.REPORTS_DIR, "02_armonizacion_columnas.md"))
    print("OK ->", out)


if __name__ == "__main__":
    main()
