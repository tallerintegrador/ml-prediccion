# -*- coding: utf-8 -*-
"""
run_all.py — Orquestador del pipeline EDA v2. Ejecuta los scripts en orden.
Uso:  python src/run_all.py
"""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEPS = [
    "00_inventario.py",
    "01_carga.py",
    "02_target.py",
    "03_calidad.py",
    "04_univar_bivar.py",
    "05_temporal.py",
    "06_multivariado.py",
    "07_diagnostico.py",
]

if __name__ == "__main__":
    for s in STEPS:
        print("\n" + "=" * 70 + f"\n>>> {s}\n" + "=" * 70)
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    print("\nPipeline completo. Ver reports/ y reports/figures/.")
