# -*- coding: utf-8 -*-
"""
config.py — Configuración central del EDA v2 (Hortifrut costos de importación).

Toda la armonización se RECONSTRUYE desde cero a partir de la inspección de los
8 CSV crudos (no se reutilizan los .xlsx de legacy/). Aquí se centralizan:
  - rutas
  - tokens de nulo
  - configuración por archivo (separador, encoding, columna de ID, etiqueta de esquema)
  - mapa canónico de columnas  (canónico -> variantes normalizadas observadas en los CSV)
  - metadatos por variable canónica (rol, disponibilidad al predecir, importancia)
  - reglas de canonicalización de conceptos de costo (regex sobre el texto normalizado)
  - política del target (qué conceptos suman al costo logístico y cuáles se excluyen)
"""
import os

SEED = 42

# --------------------------------------------------------------------------- #
# Rutas
#   Este módulo vive en src/lib/ -> ROOT sube 3 niveles (lib -> src -> raíz).
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data_csv", "raw")
PROC_DIR = os.path.join(ROOT, "data_csv", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

# reports/ se separa por etapa del pipeline
REPORTS_DIR = os.path.join(ROOT, "reports")          # raíz (compat)
REPORTS_EDA = os.path.join(REPORTS_DIR, "eda")       # informes 00..13 + figuras EDA
REPORTS_ML = os.path.join(REPORTS_DIR, "modelado")   # informe 14, métricas, predicciones
FIG_EDA = os.path.join(REPORTS_EDA, "figures")
FIG_ML = os.path.join(REPORTS_ML, "figures")
FIG_DIR = FIG_EDA                                    # alias compat (EDA)

for _d in (PROC_DIR, MODELS_DIR, REPORTS_EDA, REPORTS_ML, FIG_EDA, FIG_ML):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Tokens de nulo heterogéneos (sección 4.1 / advertencias del enunciado)
# --------------------------------------------------------------------------- #
NULL_TOKENS = ["", " ", "NA", "N/A", "n/a", "na", "-", "--", "---", ".", "..",
               "#N/D", "#N/A", "#¡REF!", "#REF!", "#VALUE!", "#¡VALOR!",
               "PEND", "PENDIENTE", "pendiente", "null", "NULL", "None", "none",
               "s/d", "S/D", "x", "X", "?", "N.A.", "n.d.", "ND"]

# --------------------------------------------------------------------------- #
# Configuración por archivo operativo (descubierta en 00_inventario)
#   id_col = columna que contiene el identificador de operación en ese esquema
#   schema = etiqueta del esquema para el reporte de deriva
# --------------------------------------------------------------------------- #
OPERATIVE_FILES = {
    "report_importaciones_2019-2020.csv": dict(sep=",", enc="utf-8-sig", id_col="REF",          schema="2019-2020"),
    "report_importaciones_2021.csv":      dict(sep=";", enc="utf-8-sig", id_col="OP",           schema="2021"),
    "report_importaciones_2022.csv":      dict(sep=",", enc="utf-8-sig", id_col="OP",           schema="2022"),
    "report_importaciones_2023.csv":      dict(sep=",", enc="utf-8-sig", id_col="OP",           schema="2023"),
    "report_importaciones_2024.csv":      dict(sep=",", enc="utf-8-sig", id_col="OPERATION",    schema="2024"),
    "report_importaciones_2025.csv":      dict(sep=",", enc="utf-8-sig", id_col="OP REFERENCE", schema="2025"),
    "report_importaciones_2026.csv":      dict(sep=",", enc="utf-8-sig", id_col="OP REFERENCE", schema="2026"),
}
EXPENSE_FILE = "bd_expense_report_importaciones_201X.csv"
EXPENSE_CFG = dict(sep=",", enc="utf-8-sig", id_col="Nro. Ope.")

# --------------------------------------------------------------------------- #
# Mapa canónico de columnas operativas.
#   canónico_snake -> lista de variantes (TAL CUAL se normalizan con
#   utils.norm_header: MAYÚSCULAS, sin acentos, espacios colapsados, sin '.').
# Las columnas no mapeadas se conservan con prefijo 'raw__' y se reportan como huérfanas.
# --------------------------------------------------------------------------- #
CANON_COLUMNS = {
    # --- Identificadores ---
    "op_id":            ["REF", "OP", "OPERATION", "OP REFERENCE", "NRO OPE"],
    "oc":               ["OC"],
    "oc2":              ["OC2"],
    "factura":          ["FT", "FACT", "FACTURA", "INVOICE"],
    "bl_awb":           ["B/L- AWB", "BL / AWB", "B/L-AWB", "BL/AWB"],
    "dam":              ["DAM"],
    "booking":          ["BOOKING"],
    "cntr_nr":          ["CNTR NR"],
    "seal":             ["SEAL"],
    # --- Temporales de campaña / negocio ---
    "campania":         ["CAMPANA", "SEASON"],
    "status":           ["STATUS", "ESTADO"],
    "importador":       ["IMPORTADOR", "IMPORTER", "IMPORTADOR/SOC", "IMPORTADOR / SOC"],
    "buyer":            ["BUYER"],
    "supplier":         ["PROVEEDOR", "SUPPLIER", "PROVEEDOR PRINCIPAL"],
    "proyecto":         ["PROYECT", "PROYECTO", "PROJECT"],
    # --- Producto / mercadería ---
    "categoria":        ["TIPO PROD", "CATEGORY", "CATEGORIA"],
    "producto":         ["PRODUCTO", "PRODUCT"],
    "qty":              ["CANT", "QTY"],
    "um":               ["UN", "UM"],
    "moneda":           ["CUR", "MONEDA"],
    "amount":           ["AMOUNT"],
    "amount_usd":       ["VALOR FINAL USD", "AMOUNT USD"],
    "payment_term":     ["FORMA DE PAGO", "PAYMENT TERM"],
    # --- Logística / transporte ---
    "incoterm":         ["INCOTERM"],
    "mode":             ["MODE"],
    "type":             ["TYPE"],
    "country_origin":   ["COUNTRY ORIGIN"],
    "pol":              ["POL"],
    "pod":              ["POD"],
    "ctnr_qty":         ["CTNR QTY", "QTY CTNR"],
    "ctnr_type":        ["CTNR TYPE", "TYPE CTNR"],
    "bulks":            ["BULKS", "PQ"],
    "bulks_type":       ["BULKS TYPE"],
    "peso_bruto":       ["PESO BRUTO"],
    "ffw":              ["FFW", "AGENTE DE CARGA"],
    "ffw_reference":    ["FFW REFERENCE"],
    "shipping_line":    ["NAVIERA / AEROLINEA", "SHIPPING LINE"],
    "vessel":           ["NAVE", "VESSEL"],
    "depot":            ["DEPOT"],
    "customs_agent":    ["AG ADUANA", "CUSTOMS AGENT", "AGENCIA DE ADUANA"],
    "canal":            ["CANAL"],
    "insurance_hper":   ["INSURANCE HPER", "SEGURO HPER"],
    "delivery_type":    ["DELIVERY TYPE"],
    "punto_llegada":    ["PUNTO DE LLEGADA", "FINAL DELIVERY POINT", "LUGAR ENTREGA"],
    "transp_t1":        ["TRANSP LIMA (T1)", "TRANSPORTE"],
    "transp_t2":        ["TRANSP TRUX (T2)"],
    # --- Fechas del proceso (para tiempos derivados) ---
    "cargo_ready":      ["CARGO READY"],
    "quote_request_date": ["QUOTE REQUEST DATE", "QUOTE REQUEST RATE"],
    "assignment_date":  ["ASSIGMENT DATE", "ASSIGMENT DATE2", "ASSIGNMENT DATE"],
    "pick_up_date":     ["PICK UP DATE"],
    "atd":              ["ATD", "ETD", "ETD 1"],
    "ata":              ["ATA", "ETA", "ETA 1"],
    "week_ata":         ["WEEK ATA", "ARRIVAL WEEK", "WEEK"],
    "ie_date":          ["FECHA IE", "IE DATE"],
    "fecha_num_dam":    ["FECHA NUM DAM", "FECHA NUMERACION"],
    "fecha_descarga":   ["FECHA DESCARGA", "DOWNLOAD DATE"],
    "fecha_tarja":      ["FECHA TARJA", "TALLY DATE"],
    "fecha_levante":    ["FECHA LEVANTE", "CUSTOMS RELEASE DATE"],
    "senasa_inspeccion": ["FECHA INSPECCION SENASA", "SENASA INSPECTION DATE"],
    "senasa_liberacion": ["FECHA LIBERACION SENASA", "SENASA RELEASE DATE"],
    "demurrage_exp":    ["FECHA VCTO SOBRESTADIA", "DEMURRAGE EXPIRATION DATE"],
    "fecha_retiro_t1":  ["FECHA RETIRO (T1)", "WITHDRAWL DATE", "FECHA DE RETIRO Y ENTREGA AGERSA"],
    "fecha_entrega_base_t2": ["FECHA ENTREGA BASE T2 (T1)", "DELIVERY DATE BASE T2 (T1)"],
    "fecha_fin_viaje":  ["FECHA FIN DE VIAJE (T2)", "FINAL ARRIVAL DATE", "FECHA ENTREGA TRUX"],
    "receipt_confirmation": ["FECHA CONFIRMACION RECEPCION", "RECEIPT CONFIRMATION DATE"],
    # --- Contables / POSTERIORES a la llegada (riesgo de leakage) ---
    "fecha_liquidacion": ["FECHA LIQUIDACION", "FECHA ENVIO LIQUIDACION", "ENVIO DE PROVISION",
                          "FECHA ENVIO ARCHIV"],
    "fecha_registro_gastos": ["FECHA DE REGISTRO DE GASTOS"],
}

# --------------------------------------------------------------------------- #
# Metadatos por variable canónica (para el diccionario de datos y leakage).
#   rol: id | feature | leakage | target_source
#   dispo: disponibilidad al momento de la LLEGADA (Sí/No/Parcial)
#   imp: importancia esperada como driver (alta/media/baja)
# --------------------------------------------------------------------------- #
VAR_META = {
    "op_id":        ("id",      "No",      "alta",  "ID único de la operación de importación (llave de unión)."),
    "oc":           ("id",      "Sí",      "baja",  "Orden de compra SAP (identificador comercial)."),
    "campania":     ("feature", "Sí",      "alta",  "Campaña/año de la operación; captura inflación de tarifas."),
    "status":       ("feature", "Parcial", "media", "Estado de la operación; solo 'liquidada' tiene costo fiable."),
    "importador":   ("feature", "Sí",      "baja",  "Empresa importadora (razón social)."),
    "buyer":        ("feature", "Sí",      "baja",  "Comprador responsable."),
    "supplier":     ("feature", "Sí",      "media", "Proveedor de la mercadería."),
    "categoria":    ("feature", "Sí",      "alta",  "Categoría de producto (plantas, sustratos, E&E, etc.)."),
    "producto":     ("feature", "Sí",      "media", "Producto específico (alta cardinalidad)."),
    "qty":          ("feature", "Sí",      "media", "Cantidad de mercadería."),
    "um":           ("feature", "Sí",      "baja",  "Unidad de medida de la cantidad."),
    "amount":       ("feature", "Sí",      "media", "Valor de la mercadería en moneda original."),
    "amount_usd":   ("feature", "Sí",      "media", "Valor de la mercadería en USD."),
    "moneda":       ("feature", "Sí",      "baja",  "Moneda de la factura de mercadería."),
    "payment_term": ("feature", "Sí",      "baja",  "Plazo de pago (días)."),
    "incoterm":     ("feature", "Sí",      "alta",  "Incoterm: determina qué tramos asume el importador."),
    "mode":         ("feature", "Sí",      "alta",  "Modo de transporte (AIR/SEA): driver primario del costo."),
    "type":         ("feature", "Sí",      "media", "Tipo de carga (FCL/LCL/AIR)."),
    "country_origin": ("feature", "Sí",    "alta",  "País de origen de la mercadería."),
    "pol":          ("feature", "Sí",      "alta",  "Puerto/aeropuerto de origen."),
    "pod":          ("feature", "Sí",      "media", "Puerto/aeropuerto de destino."),
    "ctnr_qty":     ("feature", "Sí",      "alta",  "Nº de contenedores (driver de flete marítimo)."),
    "ctnr_type":    ("feature", "Sí",      "media", "Tipo de contenedor (40HC, 20, reefer...)."),
    "bulks":        ("feature", "Sí",      "media", "Nº de bultos/pallets."),
    "peso_bruto":   ("feature", "Parcial", "alta",  "Peso bruto (clave para costo aéreo; solo 2025-2026)."),
    "ffw":          ("feature", "Sí",      "media", "Agente de carga / freight forwarder."),
    "shipping_line": ("feature", "Sí",     "media", "Naviera / aerolínea."),
    "depot":        ("feature", "Sí",      "media", "Depósito temporal."),
    "customs_agent": ("feature", "Sí",     "media", "Agencia de aduana."),
    "canal":        ("feature", "Parcial", "media", "Canal de aforo (rojo/naranja/verde); conocido tras numeración."),
    "insurance_hper": ("feature", "Sí",    "baja",  "Flag: seguro contratado por Hortifrut."),
    "delivery_type": ("feature", "Sí",     "baja",  "Tipo de entrega/direccionamiento."),
    "punto_llegada": ("feature", "Sí",     "media", "Fundo/base de destino final."),
    # fechas -> se usan para derivar tiempos, no como features crudas
    "cargo_ready":  ("feature", "Sí",      "baja",  "Fecha mercadería lista en origen."),
    "atd":          ("feature", "Sí",      "media", "Fecha real de zarpe/despegue."),
    "ata":          ("feature", "Sí",      "alta",  "Fecha real de arribo (momento de predicción)."),
    "ie_date":      ("feature", "Sí",      "baja",  "Fecha ingreso/entrega documental."),
    "fecha_num_dam": ("feature", "Sí",     "media", "Fecha de numeración de la DAM."),
    "fecha_descarga": ("feature", "Sí",    "media", "Fecha de descarga."),
    "fecha_levante": ("feature", "Parcial", "media", "Fecha de levante aduanero."),
    "senasa_inspeccion": ("feature", "Parcial", "baja", "Fecha inspección SENASA."),
    "senasa_liberacion": ("feature", "Parcial", "baja", "Fecha liberación SENASA."),
    "demurrage_exp": ("feature", "Sí",     "media", "Vencimiento de sobrestadía."),
    "fecha_fin_viaje": ("feature", "Parcial", "baja", "Fecha de entrega final en fundo."),
    # Contables -> LEAKAGE
    "fecha_liquidacion":     ("leakage", "No", "baja", "Fecha de liquidación contable: POSTERIOR a la facturación."),
    "fecha_registro_gastos": ("leakage", "No", "baja", "Fecha de registro de gastos en SAP: POSTERIOR."),
}

# --------------------------------------------------------------------------- #
# Reglas de canonicalización de CONCEPTOS de costo (sección 4.0.5 / target).
# Se aplican EN ORDEN sobre el texto del concepto normalizado (utils.norm_text:
# minúsculas, sin acentos, espacios colapsados). Primera regex que matchea gana.
# Derivadas leyendo los 81 conceptos crudos del expense report.
# --------------------------------------------------------------------------- #
CONCEPT_RULES = [
    (r"percepcion.*igv|igv",                              "PERCEPCION_IGV"),
    (r"ad ?valorem|derecho|nacionalizac|arancel|tributo", "DERECHOS_IMPUESTOS"),
    (r"flete intern|falso flete|emision bl|gastos? en origen|flete",
                                                          "FLETE_INTERNACIONAL"),
    (r"sobrestad|sobreestad|almacenaj|demurrage|detention|pernocte",
                                                          "SOBRESTADIA_ALMACENAJE"),
    (r"descarga",                                         "DESCARGA"),
    (r"thcd|gate ?in|plataforma|precinto|prencito|estiba|traccion|handling|manejo",
                                                          "HANDLING_PUERTO"),
    # T2 (destino fundo) ANTES que T1: el destino define el tramo.
    (r"transporte.*(trujillo|chao|olmos|salaverry|viru|fundo|sullana|piura)",
                                                          "TRANSPORTE_T2_LIMA_FUNDO"),
    (r"transporte.*(callao|lima)\s*-\s*lima|callao\s*-\s*lima|transporte callao|transporte lima",
                                                          "TRANSPORTE_T1_CALLAO_LIMA"),
    (r"agenciamiento|aduana|cobro admin|uso de plataforma web|courier|mensajer",
                                                          "AGENCIAMIENTO_ADUANA"),
    (r"fitosanitar|senasa|tratamiento|laboratorio fitos|fumigac",
                                                          "FITOSANITARIOS_SENASA"),
    (r"inspeccion|verificacion|visto bueno|aforo|supervis|informe",
                                                          "INSPECCION_VERIFICACION"),
    (r"seguro|poliza",                                    "SEGUROS"),
    (r"servicio logistico|servicio extraordinario|gastos? operativ|gastos? admin|multa|rectificac",
                                                          "SERVICIOS_LOGISTICOS"),
]
CONCEPTO_CANON_DEFAULT = "OTROS_LOGISTICOS"

# Conceptos que NO entran al target de costo logístico:
#   - PERCEPCION_IGV: es IGV (excluido por política contable)
CONCEPT_EXCLUDE_TARGET = {"PERCEPCION_IGV"}
# Concepto de tributos: se reporta aparte. El target principal lo INCLUYE
# (es costo de internamiento), pero se calcula también una variante sin tributos.
CONCEPT_TAXES = {"DERECHOS_IMPUESTOS"}

# Tipo de cambio de respaldo (solo para valor de mercadería operativo cuando falta
# AMOUNT USD, p. ej. 2021). El target NO lo usa: el expense ya trae Monto Total USD.
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "PEN": 0.27}
