# -*- coding: utf-8 -*-
"""
00_inventario.py — Descubrimiento e inventario de los 8 CSV crudos.
Fase 4.0 del EDA (CRISP-DM): no se asume esquema; se inspecciona cada archivo.
Genera reports/00_inventario.md con encoding, separador, dimensiones y columnas.
"""
import os, glob, csv, io
import pandas as pd

RAW = "data_csv/raw"
OUT = "reports/00_inventario.md"
os.makedirs("reports", exist_ok=True)

NULL_TOKENS = ["", "NA", "N/A", "n/a", "-", "--", "#N/D", "#¡REF!", "#REF!",
               "PEND", "PENDIENTE", "null", "NULL", "None", "s/d", "S/D", "."]

def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read(4)
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    # try utf-8 then latin-1
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            data.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"

def detect_sep(path, enc):
    with open(path, "r", encoding=enc, errors="replace") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # fallback: count candidates in first line
        first = sample.splitlines()[0] if sample else ""
        counts = {sep: first.count(sep) for sep in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get)

lines = []
def w(s=""):
    lines.append(s)

w("# 4.0 — Inventario y descubrimiento de los archivos crudos\n")
files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
w(f"Total de archivos encontrados: **{len(files)}**\n")

summary = []
for path in files:
    name = os.path.basename(path)
    size_kb = os.path.getsize(path) / 1024
    enc = detect_encoding(path)
    sep = detect_sep(path, enc)
    # raw first 3 lines
    with open(path, "r", encoding=enc, errors="replace") as f:
        raw_lines = [next(f, "").rstrip("\n").rstrip("\r") for _ in range(3)]
    # load as strings to avoid type coercion surprises
    try:
        df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str,
                         keep_default_na=False, na_values=NULL_TOKENS,
                         engine="python", on_bad_lines="warn")
        ok = True
    except Exception as e:
        df = None; ok = False; err = str(e)

    w("\n" + "=" * 78)
    w(f"\n## {name}")
    w(f"- Tamaño: **{size_kb:.1f} KB**")
    w(f"- Encoding detectado: `{enc}`  |  Separador: `{repr(sep)}`")
    if ok:
        w(f"- Dimensiones: **{df.shape[0]:,} filas × {df.shape[1]} columnas**")
        w(f"- Columnas ({df.shape[1]}):")
        for i, c in enumerate(df.columns):
            nn = df[c].notna().sum()
            pct = 100 * nn / len(df) if len(df) else 0
            w(f"    {i:>2}. `{c}`  — no-nulos: {nn:,} ({pct:.0f}%)")
        summary.append((name, df.shape[0], df.shape[1], enc, sep))
    else:
        w(f"- ⚠️ Error al cargar: {err}")
    w("\n  Primeras 3 líneas crudas:")
    for rl in raw_lines:
        w(f"    | {rl[:300]}")

w("\n" + "=" * 78)
w("\n## Resumen comparativo\n")
w("| archivo | filas | columnas | encoding | sep |")
w("|---|---:|---:|---|---|")
for n, r, c, e, s in summary:
    w(f"| {n} | {r:,} | {c} | {e} | `{repr(s)}` |")

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("OK ->", OUT)
print(f"{len(files)} archivos inventariados")
