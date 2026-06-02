# -*- coding: utf-8 -*-
"""
run_all.py — Orquestador del pipeline completo (EDA + modelado).
Ejecuta los scripts en orden. Invocar desde la raíz del repo:

    python src/run_all.py            # todo
    python src/run_all.py eda        # solo EDA (00..07)
    python src/run_all.py ml         # solo modelado (08, 09)
"""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

EDA = [
    "eda/00_inventario.py",
    "eda/01_carga.py",
    "eda/02_target.py",
    "eda/03_calidad.py",
    "eda/04_univar_bivar.py",
    "eda/05_temporal.py",
    "eda/06_multivariado.py",
    "eda/07_diagnostico.py",
]
ML = [
    "ml/08_modelado.py",
    "ml/09_figuras.py",
]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = EDA if which == "eda" else ML if which == "ml" else EDA + ML
    for s in steps:
        print("\n" + "=" * 70 + f"\n>>> {s}\n" + "=" * 70)
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    print("\nPipeline completo. Ver reports/eda/ y reports/modelado/.")
