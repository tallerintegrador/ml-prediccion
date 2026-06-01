# -*- coding: utf-8 -*-
"""
07_diagnostico.py — Diagnóstico para el modelado (sección 4.7 + entregables 4 y 5).

  - Adecuación muestral y riesgo de sobreajuste.
  - Lista de leakage (features no conocidas al momento de la llegada).
  - Tabla de decisiones por variable (usar/derivar/descartar/excluir) con tratamiento.
  - Lista final de features candidatas por bloque.
  - Recomendaciones: transformación del target, encoding, imputación, outliers,
    validación temporal y métricas.
Genera reports/10_diagnostico.md, reports/11_features_candidatas.md,
reports/12_tabla_decisiones.md.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils as U
import dataset as D

TGT = "target_servicios"

# bloques de features candidatas (solo conocidas a la llegada)
BLOQUES = {
    "Mercadería / valor": [
        ("amount_usd", "log1p + imputar 0→mediana", "numérica, sesgada"),
        ("qty", "log1p; revisar unidades (UM)", "numérica"),
        ("categoria_canon", "one-hot (8 grupos)", "categórica baja card."),
        ("producto", "descartar como feature (alta card., usar categoría)", "alta card."),
    ],
    "Transporte / modo": [
        ("mode", "one-hot (AIR/SEA/LAND)", "driver primario"),
        ("type", "one-hot (FCL/LCL/AIR)", "categórica"),
        ("es_aereo", "binaria derivada", "flag"),
        ("ctnr_qty", "imputar 0 en aéreo; numérica", "condicional marítimo"),
        ("bulks", "log1p; imputar mediana", "driver de handling/transporte"),
        ("peso_bruto", "usar si se completa histórico; hoy solo 2025-26", "condicional"),
    ],
    "Ruta / origen": [
        ("pol", "frequency/target encoding (86 cat.)", "alta card."),
        ("pod", "one-hot (28, mayoría Callao/Lima)", "media card."),
        ("pais_origen", "imputar desde POL; one-hot top + OTROS", "53% nulo"),
        ("zona_ruta", "derivar POL→POD agrupado", "derivada"),
    ],
    "Comercial / contractual": [
        ("incoterm", "one-hot (10)", "alta importancia"),
        ("incoterm_grupo", "one-hot (4) derivada", "driver de quién paga flete"),
        ("payment_term", "numérica; imputar mediana", "numérica"),
        ("supplier", "target encoding suavizado (174)", "alta card."),
    ],
    "Proveedores logísticos": [
        ("shipping_line", "target/freq encoding (66)", "driver fuerte (ε²=0.47)"),
        ("ffw", "frequency encoding (39)", "media"),
        ("customs_agent", "one-hot (15)", "media"),
        ("depot", "frequency encoding (28)", "driver fuerte (ε²=0.46)"),
    ],
    "Temporales / tiempos": [
        ("transit_days", "ata−atd; imputar mediana por modo", "derivada"),
        ("mes_arribo", "cíclica (sin/cos) o one-hot", "estacionalidad"),
        ("anio", "ordinal; clave para validación temporal", "tendencia/inflación"),
        ("canal_rojo", "binaria; OJO sólo tras numeración DAM", "condicional"),
        ("requiere_senasa", "binaria derivada", "plantas/sustratos"),
        ("tiene_seguro", "binaria (insurance_hper)", "flag"),
    ],
}


def main():
    df = D.load(fiable_only=True)
    n = len(df)

    # ============ 4.7 diagnóstico ============
    R = U.MdReport("4.7 — Diagnóstico para el modelado")
    R.h("Adecuación muestral")
    R.p(f"- Operaciones con target fiable: **{n:,}**.")
    R.p(f"- Total joineadas (incl. parciales): 1,234; despachadas sin facturar: 196 "
        "(inferencia futura, sin target).")
    nfeat_eff = len(D.FEATURES_NUM) + len(D.FEATURES_CAT)
    R.p(f"- Features candidatas crudas: ~{nfeat_eff}; tras encoding la dimensionalidad "
        "sube (one-hot de incoterm/pol/categoría).")
    R.p(f"- **Riesgo de sobreajuste moderado**: {n:,} filas es suficiente para árboles "
        "con regularización (LightGBM), pero NO para one-hot masivo de alta cardinalidad. "
        "Usar target/frequency encoding y CV temporal. El gasto se concentra en pocos "
        "segmentos (SUSTRATOS×SEA=46%), así que conviene reportar error por segmento.")

    R.h("Riesgos de LEAKAGE (excluir como features)")
    R.p("Variables/columnas conocidas SÓLO después de la facturación o la liquidación:")
    leak_items = [
        ("Conceptos de costo desglosados", "Monto Final, Igv, Importe Total, Monto Total USD del expense — SON el target."),
        ("fecha_liquidacion / fecha_registro_gastos", "fechas contables posteriores a la facturación."),
        ("Nro Liquid., Nro. de Doc., Tipo de Doc.", "identificadores de la factura logística (no existen al predecir)."),
        ("status = LIQUIDADO/LIQUIPARCIAL", "el estado de liquidación se conoce DESPUÉS; no usar como feature."),
        ("fecha_levante / SENASA / receipt_confirmation", "parcialmente posteriores a la llegada; usar con cautela o sólo su previsión."),
        ("igv_usd, tributos_usd (como features del modelo de servicios)", "se calculan de la factura; modelar aparte, no como input."),
    ]
    R.p(pd.DataFrame(leak_items, columns=["elemento", "motivo"]).to_markdown(index=False))

    R.h("Recomendaciones de modelado")
    R.p("- **Target:** modelar `log1p(servicios)` (skew≈9 en crudo) o usar objetivo "
        "Gamma/Tweedie; back-transform con corrección de sesgo (smearing). Tributos: "
        "modelo aparte o cálculo arancel×CIF.\n"
        "- **Encoding:** one-hot para baja card. (mode, type, incoterm, categoría, pod); "
        "target/frequency encoding suavizado para alta card. (supplier, pol, shipping_line, depot, ffw).\n"
        "- **Nulos:** numéricas → mediana (por modo cuando aplique); categóricas → "
        "'DESCONOCIDO'; contenedor → 0 en aéreo (no es faltante real).\n"
        "- **Outliers:** no eliminar (operaciones caras legítimas); robustez vía log y "
        "modelos basados en árboles; winsorizar sólo errores claros de captura.\n"
        "- **Validación:** **forward chaining por campaña** (entrenar ≤año t, validar año t+1); "
        "NUNCA K-fold aleatorio (hay deriva temporal +15%/año).\n"
        "- **Métricas:** MAE y WAPE (negocio: provisión total), MdAPE (robusta al sesgo), "
        "RMSE (penaliza grandes), pinball loss + cobertura de intervalos (para la "
        "pre-liquidación con incertidumbre).")

    R.h("Datos adicionales que convendría recopilar")
    R.p("- **Peso bruto histórico** (hoy sólo 2025-2026): clave para costo aéreo y unitario.\n"
        "- **Partida arancelaria (HS) / tasa Ad Valorem**: haría los tributos calculables.\n"
        "- **Tipo de cambio por factura** y fecha: para multimoneda exacta.\n"
        "- **Tarifa de flete cotizada** (quote) al momento del embarque.\n"
        "- **Distancia/zona fundo destino** y nº real de contenedores por operación.")
    R.save(os.path.join(C.REPORTS_DIR, "10_diagnostico.md"))

    # ============ Entregable 4 — features candidatas ============
    F = U.MdReport("Entregable 4 — Features candidatas finales (conocidas a la llegada)")
    for bloque, items in BLOQUES.items():
        F.h(bloque)
        F.p(pd.DataFrame(items, columns=["feature", "tratamiento", "nota"]).to_markdown(index=False))
    F.save(os.path.join(C.REPORTS_DIR, "11_features_candidatas.md"))

    # ============ Entregable 5 — tabla de decisiones ============
    T = U.MdReport("Entregable 5 — Tabla de decisiones por variable")
    rows = []
    for c in df.columns:
        if c not in C.VAR_META and c not in ("op_id_full", "op_key"):
            continue
        meta = C.VAR_META.get(c)
        if meta is None:
            continue
        rol, dispo, imp, desc = meta
        nul = round(100 * df[c].isna().mean(), 1)
        card = df[c].nunique()
        if rol == "leakage":
            dec, trat = "EXCLUIR", "data leakage (posterior a la llegada)"
        elif rol == "id":
            dec, trat = "ID (no feature)", "llave de unión / identificador"
        elif nul > 80:
            dec, trat = "DESCARTAR", f"{nul}% nulos"
        elif c in ("ata", "atd", "fecha_num_dam", "fecha_descarga", "fecha_levante",
                   "fecha_fin_viaje", "senasa_inspeccion", "senasa_liberacion",
                   "demurrage_exp", "ie_date"):
            dec, trat = "DERIVAR", "extraer tiempos/flags, no usar fecha cruda"
        elif df[c].dtype.kind in "biufc" or c in D.FEATURES_NUM:
            dec, trat = "USAR", "numérica: imputar mediana" + (" + log" if c in ("amount_usd",) else "")
        else:
            dec, trat = "USAR", ("target/freq encoding" if card > 20 else "one-hot")
        rows.append(dict(variable=c, rol=rol, dispo=dispo, importancia=imp,
                         pct_nulos=nul, cardinalidad=card, decision=dec, tratamiento=trat))
    dt = pd.DataFrame(rows)
    order = {"USAR": 0, "DERIVAR": 1, "ID (no feature)": 2, "DESCARTAR": 3, "EXCLUIR": 4}
    dt = dt.sort_values(["decision", "importancia"], key=lambda s: s.map(order).fillna(9)
                        if s.name == "decision" else s)
    T.p(dt.to_markdown(index=False))
    T.h("Resumen")
    T.p(dt["decision"].value_counts().rename_axis("decisión").reset_index(name="n").to_markdown(index=False))
    T.save(os.path.join(C.REPORTS_DIR, "12_tabla_decisiones.md"))

    print("OK 4.7 -> 10_diagnostico.md, 11_features_candidatas.md, 12_tabla_decisiones.md")
    print("decisiones:", dt["decision"].value_counts().to_dict())


if __name__ == "__main__":
    main()
