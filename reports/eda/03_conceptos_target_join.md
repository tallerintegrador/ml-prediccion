# 4.0.5 / 4.1 — Conceptos canónicos, target y join


## Conceptos canónicos (81 crudos → 14 canónicos)

Reglas regex aplicadas en orden sobre el texto normalizado; primera que matchea gana.
| concepto_canon            |   n_lineas |        usd_total |   n_originales |   usd_pct | ejemplos                                                                                                              |
|:--------------------------|-----------:|-----------------:|---------------:|----------:|:----------------------------------------------------------------------------------------------------------------------|
| DERECHOS_IMPUESTOS        |       1514 |      1.78143e+07 |              5 |      59   | Derechos, Ad Valorem, Gastos de Nacionalización                                                                       |
| PERCEPCION_IGV            |        743 |      3.1928e+06  |              2 |      10.6 | Percepción del IGV, percepción del IGV                                                                                |
| FLETE_INTERNACIONAL       |       1050 |      3.05036e+06 |              4 |      10.1 | Flete Internacional, Gastos en Origen, Emisión BL                                                                     |
| DESCARGA                  |       1760 |      1.58187e+06 |              2 |       5.2 | Descarga, Descarga - Nota de Crédito                                                                                  |
| TRANSPORTE_T2_LIMA_FUNDO  |       1538 |      1.56036e+06 |              8 |       5.2 | Transporte Lima - Trujillo, Transporte Callao - Trujillo, Transporte Lima - Chao                                      |
| SOBRESTADIA_ALMACENAJE    |        494 | 869210           |              7 |       2.9 | Sobrestadía, Gastos Admin. Sobrestadía, Pernocte                                                                      |
| SERVICIOS_LOGISTICOS      |       1555 | 664167           |             10 |       2.2 | Gastos Operativos, Servicio Logistico Integral, Servicio Extraordinario                                               |
| TRANSPORTE_T1_CALLAO_LIMA |        868 | 398933           |              2 |       1.3 | Transporte Callao - Lima, Transporte Callao - Callao                                                                  |
| HANDLING_PUERTO           |       1582 | 319118           |             10 |       1.1 | Handling, THCD, Manejo de Plataformas                                                                                 |
| AGENCIAMIENTO_ADUANA      |       1052 | 297354           |              5 |       1   | Agenciamiento de Aduana, Cobro Administrativo, Agenciamiento de Aduana - Nota de Crédito                              |
| INSPECCION_VERIFICACION   |        903 | 264460           |              3 |       0.9 | Informe de Inspeccion y Verificación, Visto Bueno, Aforo                                                              |
| FITOSANITARIOS_SENASA     |       1132 |  65987           |              8 |       0.2 | Permiso Fitosanitario de Importación, Analisis de Laboratorio Fitosanitario, Supervisión de Tratamiento Fitosanitario |
| SEGUROS                   |        645 |  56911           |              1 |       0.2 | Poliza de Seguro                                                                                                      |
| OTROS_LOGISTICOS          |         43 |  41949           |             14 |       0.1 | Daño de Contenedor, Uso de Área Operativa, Servicio de Trazegado                                                      |

## Duplicados detectados por normalización (mayúsculas/acentos/espacios)

- `derechos` ← ['Derechos', 'Derechos ', 'derechos']
- `fumigacion` ← ['Fumigacion', 'Fumigación']
- `gate in` ← ['GATE IN', 'Gate In']
- `inspeccion fitosanitaria de importacion` ← ['Inspeccion fitosanitaria de importacion', 'Inspección fitosanitaria de importación']
- `multa` ← ['MULTA', 'Multa', 'multa']
- `percepcion del igv` ← ['Percepción del IGV', 'percepción del IGV']
- `pernocte` ← ['Pernocte', 'Pernocte ']
- `servicio extraordinario` ← ['Servicio Extraordinario', 'servicio extraordinario']

## Reconstrucción del target

Llave de unión: **op_id_full** (identificador completo normalizado, p. ej. `20119HPER`). Elegida sobre la llave laxa `YY-NNN` porque ésta colapsaba **108** operaciones distintas (mismo año+secuencia, distinto código de empresa).
- Operaciones únicas en EXPENSE: **1,276**
- Operaciones únicas en OPERATIVO: **1,430**
- **Join exitoso (ambos): 1,234** (96.7% del expense)
- Solo operativo (despachadas, sin gasto cargado aún): **196**  ← escenario real de predicción
- Solo expense (huérfanas, sin operativo — sobre todo formato legacy 2019): **42**

**Granularidad final:** 1 fila por operación. Dataset modelable = **1,234 operaciones × 73 columnas**.
- Promedio de conceptos por operación: 7.7
- Promedio de líneas de costo por operación: 11.1
- Colisiones con op_id_full: **0** (vs 108 con YY-NNN).

## Estadísticos del target (USD)

|                     |   count |    mean |     std |   min |    5% |    25% |     50% |     75% |     95% |    max |   skew |   kurtosis |
|:--------------------|--------:|--------:|--------:|------:|------:|-------:|--------:|--------:|--------:|-------:|-------:|-----------:|
| target_usd          |    1234 | 21627.6 | 28668.2 |  10   | 238.8 | 3721.5 | 13878.2 | 27686.1 | 77024.4 | 309798 |    3.7 |       22.1 |
| target_usd_sin_trib |    1211 |  7517.2 | 11667.7 |  16.8 | 183.7 | 1671.5 |  4835.6 |  9148.9 | 24008.1 | 241365 |    9   |      146.7 |
| igv_usd             |     719 |  4414.5 |  6959.2 |   0   |  86.5 |  943.2 |  2275.4 |  4350   | 19021.8 |  74064 |    4   |       22.8 |

- Operaciones con target_usd > 0: 1,234 / 1,234
- Operaciones con target ≤ 0 o nulo: 0 (revisar)

## Cobertura del target por estado (status)

| status       |   n |   con_target |   target_medio |
|:-------------|----:|-------------:|---------------:|
| LIQUIDADO    | 670 |          670 |          12560 |
| LIQUIDADA    | 261 |          261 |          18318 |
| LIQUIPARCIAL | 145 |          145 |          19233 |
| ARCHIVADA    | 127 |          127 |           9126 |
| LIQPARCIAL   |  16 |           16 |           2558 |
| RETIRADO     |   6 |            6 |           3941 |
| ADUANAS      |   3 |            3 |           1752 |
| POR RETIRAR  |   2 |            2 |          10512 |
| ORIGEN       |   1 |            1 |           1880 |
| LIQUIPARCIAl |   1 |            1 |          61679 |
| RECIBIDO     |   1 |            1 |           2963 |
| TRANSITO     |   1 |            1 |           6442 |

## Cobertura del target por campaña

|   campania_exp |   n |   target_medio |    target_total |
|---------------:|----:|---------------:|----------------:|
|           2020 | 118 |           8777 |     2.48775e+06 |
|           2021 | 269 |          17553 |     7.58156e+06 |
|           2022 | 267 |          17803 |     6.72068e+06 |
|           2023 | 150 |           4891 |     2.58425e+06 |
|           2024 | 225 |          14180 |     4.3581e+06  |
|           2025 | 201 |           9314 |     2.91232e+06 |
|           2026 |   4 |           5695 | 43784           |