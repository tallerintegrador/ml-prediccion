# -*- coding: utf-8 -*-
"""Construye notebooks/EDA_presentacion.ipynb (deliverable 'notebook de presentación').
El notebook importa los módulos de src/ y muestra tablas y figuras ya generadas.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "EDA_presentacion.ipynb")


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": list(src)}


def code(*src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(src)}


cells = [
    md("# EDA v2 — Costos de importación Hortifrut Perú\n",
       "**Notebook de presentación.** Los cálculos viven en `src/00…07_*.py` (fuente de "
       "verdad, reproducibles con `python src/run_all.py`); aquí se cargan los resultados "
       "ya generados y se muestran tablas y figuras clave.\n\n",
       "Metodología CRISP-DM · target validado = **servicios logísticos** (tributos aparte) "
       "sobre **target fiable**."),
    code("import os, sys\n",
         "sys.path.insert(0, os.path.abspath('../src'))\n",
         "import pandas as pd, numpy as np\n",
         "from IPython.display import Image, Markdown, display\n",
         "import dataset as D, config as C\n",
         "FIG = '../reports/figures'\n",
         "df = D.load(fiable_only=True)\n",
         "print('Operaciones fiables:', df.shape)"),

    md("## 4.0 Inventario y armonización\n",
       "8 CSV (1 expense + 7 operativos) con deriva de esquema español→inglés. "
       "Ver `reports/00_inventario.md` y `reports/02_armonizacion_columnas.md`."),
    code("op = pd.read_parquet('../data_csv/processed/operativo_lineas.parquet')\n",
         "exp = pd.read_parquet('../data_csv/processed/expense_lineas.parquet')\n",
         "print('operativo_lineas:', op.shape, '| expense_lineas:', exp.shape)\n",
         "print('esquemas:', sorted(op[\"schema\"].unique()))"),

    md("## 4.0.5 / 4.1 Conceptos canónicos y reconstrucción del target\n",
       "81 conceptos → 14 canónicos; join por `op_id_full` (0 colisiones, 98.4%)."),
    code("g = (exp.assign(cc=exp['concepto_raw'].map(__import__('utils').canon_concept))\n",
         "        .groupby('cc')['monto_usd'].agg(['size','sum']).sort_values('sum', ascending=False))\n",
         "g['usd_pct'] = (100*g['sum']/g['sum'].sum()).round(1); display(g.round(0))\n",
         "display(df[['target_servicios','tributos_usd','target_total']].describe().round(0))"),
    code("Image(f'{FIG}/f02_target_dist.png')"),
    code("Image(f'{FIG}/f02_conceptos.png')"),

    md("## 4.3 / 4.4 Distribución del target y drivers\n",
       "Target asimétrico (skew≈9) → log. Drivers por ε² (Kruskal-Wallis)."),
    code("Image(f'{FIG}/f04_target.png')"),
    code("Image(f'{FIG}/f05_drivers_cat.png')"),
    code("Image(f'{FIG}/f05_escala.png')"),
    code("Image(f'{FIG}/f05_corr.png')"),

    md("## 4.5 Tiempos del proceso y estacionalidad\n",
       "Más días en depósito ⇒ más sobrestadía (ρ=0.53). Deriva temporal ≈ +15%/año "
       "⇒ validación temporal."),
    code("Image(f'{FIG}/f06_temporal.png')"),

    md("## 4.6 Segmentos, clustering e importancia preliminar\n",
       "SUSTRATOS×SEA = 46% del gasto. Baseline LightGBM (split temporal): MdAPE 24.9%."),
    code("Image(f'{FIG}/f07_segmentos.png')"),
    code("Image(f'{FIG}/f07_clusters.png')"),
    code("Image(f'{FIG}/f07_importancia.png')"),

    md("## 4.7 Diagnóstico y entregables\n",
       "- Diccionario: `reports/05_diccionario_datos.md`\n",
       "- Features candidatas: `reports/11_features_candidatas.md`\n",
       "- Tabla de decisiones: `reports/12_tabla_decisiones.md`\n",
       "- **Hallazgos + 9 preguntas:** `reports/13_hallazgos.md`"),
    code("print(open('../reports/13_hallazgos.md', encoding='utf-8').read()[:1500])"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.14"}},
      "nbformat": 4, "nbformat_minor": 5}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK ->", OUT, f"({len(cells)} celdas)")
