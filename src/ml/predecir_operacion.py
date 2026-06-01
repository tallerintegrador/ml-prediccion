# -*- coding: utf-8 -*-
"""
predecir_operacion.py — PRUEBA MANUAL: tú colocas los datos de una operación y el
modelo devuelve la aproximación del COSTO DE SERVICIOS logísticos (sin tributos), en USD,
con su intervalo P10-P90.

Uso:
  # interactivo (te pregunta campo por campo; Enter = dejar vacío/usar default):
  python src/ml/predecir_operacion.py

  # con un ejemplo precargado (sin preguntar nada):
  python src/ml/predecir_operacion.py --ejemplo

  # desde un JSON (un dict, o lista de dicts para varias operaciones):
  python src/ml/predecir_operacion.py --json mi_operacion.json

Recuerda: predice SOLO la parte variable (flete, descarga, transporte, handling...).
NO incluye tributos (Ad Valorem/IGV), que son cálculo determinístico aparte.
"""
import sys
import json
import argparse

import _predecir_lib as P


# Campos que se piden en modo interactivo: (clave, etiqueta, tipo)
#   tipo: "txt" texto | "num" número | "flag" 0/1
CAMPOS = [
    ("mode",          "Modo de transporte (SEA/AIR)",                "txt"),
    ("type",          "Tipo de carga (FCL/LCL/AIR)",                 "txt"),
    ("incoterm",      "Incoterm (FOB/CIF/EXW/DAP...)",               "txt"),
    ("categoria",     "Categoría de producto (PLANTAS/SUSTRATOS...)", "txt"),
    ("amount_usd",    "Valor de la mercadería en USD",               "num"),
    ("ctnr_qty",      "Nº de contenedores (0 si aéreo)",             "num"),
    ("qty",           "Cantidad de mercadería",                      "num"),
    ("bulks",         "Nº de bultos/pallets",                        "num"),
    ("pol",           "Puerto/aeropuerto de ORIGEN (POL)",           "txt"),
    ("pod",           "Puerto/aeropuerto de DESTINO (POD)",          "txt"),
    ("pais_origen",   "País de origen",                              "txt"),
    ("supplier",      "Proveedor",                                   "txt"),
    ("shipping_line", "Naviera / aerolínea",                         "txt"),
    ("customs_agent", "Agencia de aduana",                           "txt"),
    ("ffw",           "Agente de carga (freight forwarder)",         "txt"),
    ("depot",         "Depósito temporal",                           "txt"),
    ("punto_llegada", "Fundo / punto de llegada final",              "txt"),
    ("buyer",         "Comprador (buyer)",                           "txt"),
    ("payment_term",  "Plazo de pago (días)",                        "num"),
    ("transit_days",  "Días de tránsito (ATA - ATD)",                "num"),
    ("n_oc_distintas", "Nº de órdenes de compra distintas",          "num"),
    ("n_productos",   "Nº de productos distintos",                   "num"),
    ("requiere_senasa", "¿Requiere inspección SENASA? (1/0)",        "flag"),
    ("tiene_seguro",  "¿Tiene seguro? (1/0)",                        "flag"),
    ("anio",          "Año de arribo (ej. 2025)",                    "num"),
    ("mes",           "Mes de arribo (1-12)",                        "num"),
]

# operación de ejemplo (marítimo FCL típico) para --ejemplo
EJEMPLO = {
    "mode": "SEA", "type": "FCL", "incoterm": "FOB", "categoria": "SUSTRATOS",
    "amount_usd": 45000, "ctnr_qty": 2, "qty": 1800, "bulks": 40,
    "pol": "SHANGHAI", "pod": "CALLAO", "pais_origen": "CHINA",
    "supplier": "PROVEEDOR DEMO", "shipping_line": "MAERSK",
    "customs_agent": "AGENCIA DEMO", "ffw": "FFW DEMO", "depot": "NEPTUNIA",
    "punto_llegada": "FUNDO TRUJILLO", "buyer": "BUYER DEMO",
    "payment_term": 30, "transit_days": 38, "n_oc_distintas": 1, "n_productos": 3,
    "requiere_senasa": 1, "tiene_seguro": 1, "anio": 2025, "mes": 6,
}


def _pedir_interactivo():
    print("=" * 64)
    print(" PREDICCIÓN DE COSTO DE SERVICIOS LOGÍSTICOS (USD)")
    print(" Enter en blanco = dejar vacío (se imputa automáticamente).")
    print("=" * 64)
    datos = {}
    for clave, etiqueta, tipo in CAMPOS:
        raw = input(f"  {etiqueta}: ").strip()
        if raw == "":
            continue
        if tipo in ("num", "flag"):
            try:
                datos[clave] = float(raw)
            except ValueError:
                print(f"    (valor no numérico, se ignora '{clave}')")
        else:
            datos[clave] = raw
    return datos


def _mostrar(datos, res):
    r = res.iloc[0]
    print("\n" + "-" * 64)
    print(" RESULTADO")
    print("-" * 64)
    print(f"  Costo de servicios estimado : USD {r['servicios_pred_usd']:>12,.0f}")
    print(f"  Banda de incertidumbre P10  : USD {r['p10_usd']:>12,.0f}")
    print(f"  Banda de incertidumbre P90  : USD {r['p90_usd']:>12,.0f}")
    print("-" * 64)
    print("  Nota: NO incluye tributos (Ad Valorem/IGV). El intervalo P10-P90")
    print("  cubre la incertidumbre (sobrestadía, demoras). Error mediano ~25%.")
    print("-" * 64)


def main():
    ap = argparse.ArgumentParser(description="Predicción manual de costo de servicios.")
    ap.add_argument("--ejemplo", action="store_true", help="usa la operación de ejemplo")
    ap.add_argument("--json", metavar="ARCHIVO", help="lee la(s) operación(es) de un JSON")
    args = ap.parse_args()

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            datos = json.load(f)
    elif args.ejemplo:
        datos = EJEMPLO
        print("Usando operación de EJEMPLO:")
        print(json.dumps(datos, indent=2, ensure_ascii=False))
    else:
        datos = _pedir_interactivo()
        if not datos:
            print("No ingresaste ningún dato. Usa --ejemplo para ver una demo.")
            sys.exit(0)

    art = P.load_artifacts()
    df_raw = P.construir_fila(datos)
    res = P.predecir(art, df_raw)

    if len(res) == 1:
        _mostrar(datos, res)
    else:
        # varias operaciones (JSON con lista)
        import pandas as pd
        pd.set_option("display.max_columns", None, "display.width", 160)
        print(res.to_string(index=True))
        print(f"\nSuma servicios estimada: USD {res['servicios_pred_usd'].sum():,.0f}")


if __name__ == "__main__":
    main()
