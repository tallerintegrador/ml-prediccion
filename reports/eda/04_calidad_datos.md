# 4.2 — Calidad de datos (dataset modelable: 1 fila/operación)


## % de nulos por columna (ordenado)

| columna               |   pct_nulos |   n_unicos | tipo       | clasificación   |
|:----------------------|------------:|-----------:|:-----------|:----------------|
| fecha_registro_gastos |        97.7 |         18 | fecha      | descartable     |
| peso_bruto            |        90.9 |         85 | numérico   | descartable     |
| ffw_reference         |        89.8 |        124 | categórico | descartable     |
| cargo_ready           |        82.1 |        159 | fecha      | descartable     |
| delivery_type         |        82.1 |          3 | categórico | descartable     |
| bulks_type            |        82   |          2 | categórico | descartable     |
| quote_request_date    |        80.3 |        165 | fecha      | descartable     |
| pick_up_date          |        77.2 |        232 | fecha      | condicional     |
| importador            |        71.7 |          4 | categórico | condicional     |
| canal                 |        71.6 |          4 | categórico | condicional     |
| demurrage_exp         |        70.7 |        169 | fecha      | condicional     |
| assignment_date       |        66.4 |        264 | fecha      | condicional     |
| proyecto              |        62.8 |         17 | categórico | condicional     |
| senasa_inspeccion     |        62.4 |        218 | fecha      | condicional     |
| senasa_liberacion     |        61.9 |        228 | fecha      | condicional     |
| fecha_liquidacion     |        58.3 |         73 | fecha      | condicional     |
| receipt_confirmation  |        57.2 |        318 | fecha      | condicional     |
| country_origin        |        53.1 |         23 | categórico | condicional     |
| ctnr_qty              |        46.6 |         23 | numérico   | condicional     |
| ctnr_type             |        45.9 |          6 | categórico | condicional     |
| igv_usd               |        41.7 |        639 | numérico   | condicional     |
| fecha_tarja           |        38.6 |        439 | fecha      | condicional     |
| fecha_descarga        |        37.1 |        455 | fecha      | condicional     |
| fecha_entrega_base_t2 |        30.7 |        527 | fecha      | condicional     |
| transp_t1             |        27.1 |         21 | categórico | usable          |
| fecha_levante         |        22.7 |        577 | fecha      | usable          |
| ie_date               |        22   |        517 | fecha      | usable          |
| transp_t2             |        19.4 |         28 | categórico | usable          |
| fecha_retiro_t1       |        17.8 |        620 | fecha      | usable          |
| fecha_fin_viaje       |        14.8 |        644 | fecha      | usable          |
| vessel                |        14.6 |        517 | categórico | usable          |
| fecha_num_dam         |        14.2 |        615 | fecha      | usable          |
| moneda                |        10.4 |          2 | categórico | usable          |
| campania              |        10.3 |          6 | categórico | usable          |
| depot                 |         5.4 |         28 | categórico | usable          |
| punto_llegada         |         5.3 |         40 | categórico | usable          |
| payment_term          |         3.9 |         13 | numérico   | usable          |
| ffw                   |         2.5 |         39 | categórico | usable          |
| shipping_line         |         2.4 |         66 | categórico | usable          |
| insurance_hper        |         1.9 |          3 | categórico | usable          |
| target_usd_sin_trib   |         1.9 |       1174 | numérico   | usable          |
| customs_agent         |         0.7 |         15 | categórico | usable          |
| um                    |         0.4 |         21 | categórico | usable          |
| atd                   |         0.3 |        748 | fecha      | usable          |
| pol                   |         0.3 |         86 | categórico | usable          |
| ata                   |         0.3 |        708 | fecha      | usable          |
| mode                  |         0.2 |          3 | categórico | usable          |
| pod                   |         0.1 |         28 | categórico | usable          |
| incoterm              |         0.1 |         10 | categórico | usable          |
| type                  |         0.1 |          7 | categórico | usable          |
| amount_usd            |         0   |        906 | numérico   | usable          |
| bulks                 |         0   |        153 | numérico   | usable          |
| qty                   |         0   |        499 | numérico   | usable          |
| amount                |         0   |        785 | numérico   | usable          |
| status                |         0   |         12 | categórico | usable          |
| categoria             |         0   |         16 | categórico | usable          |
| supplier              |         0   |        174 | categórico | usable          |
| producto              |         0   |        308 | categórico | usable          |
| buyer                 |         0   |         27 | categórico | usable          |
| anio_op               |         0   |          7 | numérico   | usable          |
| n_oc_lineas           |         0   |         10 | numérico   | usable          |
| schema                |         0   |          7 | categórico | usable          |
| source_file           |         0   |          7 | categórico | usable          |
| n_productos           |         0   |          5 | numérico   | usable          |
| n_oc_distintas        |         0   |          8 | numérico   | usable          |
| target_usd            |         0   |       1203 | numérico   | usable          |
| n_conceptos           |         0   |         13 | numérico   | usable          |
| campania_exp          |         0   |          7 | categórico | usable          |
| n_lineas_costo        |         0   |         33 | numérico   | usable          |
| tributos_usd          |         0   |        937 | numérico   | usable          |
| target_fiable         |         0   |          2 | numérico   | usable          |

## Resumen por clasificación

| clasificación   |   n_columnas |
|:----------------|-------------:|
| usable          |           47 |
| condicional     |           17 |
| descartable     |            7 |

- **usable** (<30% nulos): listas para modelar.
- **condicional** (30–80%): aplican sólo a un subconjunto (p. ej. datos de contenedor sólo en marítimo, peso bruto sólo 2025-2026, fechas SENASA sólo plantas).
- **descartable** (>80%): demasiado vacías para usarse crudas.

## Patrón de ausencia: contenedor vs modo de transporte

- `ctnr_qty` nulo en **AIR**: 100%  |  en **SEA**: 10%
Confirma que los datos de contenedor faltan estructuralmente en aéreo (no es error).

## Inconsistencias detectadas

**Cardinalidad y variantes categóricas** (crudos vs normalizados: la diferencia son duplicados por mayúsculas/acentos/espacios):
| columna        |   n_crudos |   n_normalizados |   variantes_por_formato | ejemplos                                                                                   |
|:---------------|-----------:|-----------------:|------------------------:|:-------------------------------------------------------------------------------------------|
| mode           |          3 |                3 |                       0 | SEA, AIR, LAND                                                                             |
| type           |          7 |                7 |                       0 | LCL/LCL, FCL/FCL, AIR, FCL/LCL, LAND, FCL, LCL                                             |
| incoterm       |         10 |               10 |                       0 | CIF, FOB, CPT, EXW, DAP, CIP, FCA, DDP                                                     |
| categoria      |         16 |               14 |                       2 | EQUIPO E INSTRUMENTACION, SUSTRATOS, MATERIAL DE EMPAQUE, PLANTAS, UTILES DE OFICINA, AGRO |
| status         |         12 |               11 |                       1 | ARCHIVADA, LIQUIDADA, LIQUIDADO, LIQUIPARCIAL, LIQUIPARCIAl, RECIBIDO, POR RETIRAR, RETIRA |
| moneda         |          2 |                2 |                       0 | EUR, USD                                                                                   |
| country_origin |         23 |               23 |                       0 | CHILE, ESPAÑA, PAISES BAJOS, USA, CHINA, ISRAEL, MEXICO, SRI LANKA                         |
| pol            |         86 |               86 |                       0 | CALLAO, PAITA, LIMA, MANZANILLO, LEIXOES, RIGA, ALGECIRAS, BARCELONA                       |
| pod            |         28 |               28 |                       0 | BARCELONA, ESTONIA, Vlissingen, DOVER, SANTIAGO, MANZANILLO, PORTLAND, COLOMBO             |
| um             |         21 |               19 |                       2 | UN, BIG BAGS, UNI,  KGS , un, LT, KGS, LTS                                                 |

## Coherencia temporal

- Tránsito ata−atd: válidos=1229, **imposibles (<0)=9**, mediana=12 días, p95=80 días
- Días hasta levante (levante−ata): mediana=0, imposibles(<0)=77

## Montos

- `amount_usd`: ≤0 = 12, nulos = 0, negativos = 0
- `target_usd`: ≤0 = 0, nulos = 0, negativos = 0
- Monedas de mercadería presentes: ['eur', 'usd']