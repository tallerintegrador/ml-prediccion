# -*- coding: utf-8 -*-
"""
dataset.py — Carga preparada del dataset modelable para las secciones 4.3–4.7.

Centraliza las decisiones validadas con negocio:
  - TARGET PRIMARIO = servicios logísticos (excluye tributos e IGV).
  - TRIBUTOS (Ad Valorem/derechos) se conservan aparte (se modelan por separado).
  - Se entrena/analiza SOLO sobre target fiable (operaciones liquidadas/archivadas).
Además armoniza categóricas con deriva semántica y deriva tiempos del proceso.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils as U


def _categoria_canon(x):
    t = U.norm_text(x)
    if t == "":
        return np.nan
    if "sustrato" in t:                                   return "SUSTRATOS"
    if t in ("e & e", "e&e") or "empaque" in t or "envase" in t:  return "EMPAQUE"
    if "eqp" in t or "equipo" in t or "instrument" in t:  return "EQUIPOS_INSTR"
    if "sparepart" in t or "repuesto" in t:               return "REPUESTOS"
    if "planta" in t:                                     return "PLANTAS"
    if "agq" in t or "agroq" in t or t == "agro":         return "AGROQUIMICOS"
    if "maquinar" in t:                                   return "MAQUINARIA"
    if "muestra" in t:                                    return "MUESTRAS"
    return "OTROS"


# Grupo de incoterm según quién asume el flete principal (driver de costo)
_INCOTERM_GRP = {
    "EXW": "comprador_todo", "FCA": "comprador_flete", "FOB": "comprador_flete",
    "FAS": "comprador_flete",
    "CFR": "vendedor_flete", "CIF": "vendedor_flete", "CPT": "vendedor_flete",
    "CIP": "vendedor_flete",
    "DAP": "vendedor_destino", "DPU": "vendedor_destino", "DDP": "vendedor_destino",
}


def _clean_cat(s):
    return s.astype("string").str.strip().str.upper().replace({"": pd.NA})


def _safe_days(a, b):
    d = (a - b).dt.days
    return d.where(d >= 0)   # descarta imposibles (<0)


def load(fiable_only=True, add_features=True):
    m = pd.read_parquet(os.path.join(C.PROC_DIR, "operaciones_modelables.parquet"))

    # --- targets (decisión validada) ---
    m["target_total"] = m["target_usd"]
    m["tributos_usd"] = m["tributos_usd"].fillna(0.0)
    # servicios = total − tributos (siempre definido; 0 si la op solo tiene derechos)
    m["target_servicios"] = (m["target_total"] - m["tributos_usd"]).clip(lower=0)
    m["log_target_servicios"] = np.log1p(m["target_servicios"])
    m["log_target_total"] = np.log1p(m["target_total"])

    # --- limpieza categórica + armonización semántica ---
    for c in ["mode", "type", "incoterm", "country_origin", "pol", "pod",
              "supplier", "customs_agent", "ffw", "shipping_line", "depot",
              "status", "buyer", "punto_llegada"]:
        if c in m:
            m[c] = _clean_cat(m[c])
    m["categoria_canon"] = m["categoria"].map(_categoria_canon).astype("string")
    m["incoterm_grupo"] = m["incoterm"].map(_INCOTERM_GRP).astype("string")
    # país de origen: imputar faltante desde POL conocido (heurística simple)
    m["pais_origen"] = m["country_origin"]

    if add_features:
        # --- tiempos del proceso (4.5) ---
        m["transit_days"] = _safe_days(m["ata"], m["atd"])
        m["dias_levante"] = _safe_days(m["fecha_levante"], m["ata"])
        m["dias_num_dam"] = _safe_days(m["fecha_num_dam"], m["ata"])
        m["dias_a_fundo"] = _safe_days(m["fecha_fin_viaje"], m["ata"])
        m["dias_deposito"] = _safe_days(m["fecha_retiro_t1"], m["fecha_descarga"])
        # --- temporales de arribo ---
        m["anio"] = m["anio_op"].astype("Int64")
        m["mes_arribo"] = m["ata"].dt.month
        m["trimestre_arribo"] = m["ata"].dt.quarter
        # --- costo unitario (más estable que el absoluto) ---
        m["costo_unit_peso"] = np.where(m["peso_bruto"] > 0,
                                        m["target_servicios"] / m["peso_bruto"], np.nan)
        m["costo_unit_valor"] = np.where(m["amount_usd"] > 0,
                                         m["target_servicios"] / m["amount_usd"], np.nan)
        m["ratio_flete_valor"] = m["costo_unit_valor"]
        # --- flags de negocio ---
        m["es_aereo"] = (m["mode"] == "AIR").fillna(False).astype(int)
        m["canal_rojo"] = m["canal"].map(U.norm_text).str.contains("rojo", na=False).astype(int)
        m["requiere_senasa"] = m["senasa_inspeccion"].notna().astype(int)
        m["tiene_seguro"] = m["insurance_hper"].map(
            lambda x: 1 if U.norm_text(x).startswith("si") else 0)

    if fiable_only:
        m = m[m["target_fiable"].astype(bool)].copy()
    return m


# columnas feature candidatas conocidas a la llegada (sin leakage)
FEATURES_NUM = ["amount_usd", "qty", "bulks", "ctnr_qty", "peso_bruto",
                "payment_term", "transit_days", "n_oc_distintas", "n_productos"]
FEATURES_CAT = ["mode", "type", "incoterm", "incoterm_grupo", "categoria_canon",
                "pais_origen", "pol", "pod", "supplier", "customs_agent", "ffw",
                "shipping_line", "depot", "punto_llegada", "buyer"]


if __name__ == "__main__":
    df = load()
    print("fiable:", df.shape)
    print(df[["target_servicios", "tributos_usd", "target_total"]].describe().round(0))
    print("\ncategoria_canon:\n", df["categoria_canon"].value_counts())
