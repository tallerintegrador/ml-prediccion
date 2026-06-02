# 4.0.3 / 4.1 — Armonización de columnas (deriva de esquema)

Mapa canónico reconstruido **desde cero** inspeccionando los 7 esquemas operativos. Cada fila es una variable canónica; cada columna, el nombre crudo real en ese archivo (— = ausente).

## Matriz de armonización canónica × esquema

| canónica              | 2019-2020                        | 2021                       | 2022                         | 2023                         | 2024                         | 2025                        | 2026                        |
|:----------------------|:---------------------------------|:---------------------------|:-----------------------------|:-----------------------------|:-----------------------------|:----------------------------|:----------------------------|
| op_id                 | REF                              | OP                         | OP                           | OP                           | OPERATION                    | OP REFERENCE                | OP REFERENCE                |
| oc                    | OC                               | OC                         | OC                           | OC                           | OC                           | OC                          | OC                          |
| oc2                   | —                                | —                          | —                            | OC2                          | OC2                          | OC2                         | OC2                         |
| factura               | FT                               | FACT                       | FACTURA                      | FACTURA                      | INVOICE                      | INVOICE                     | INVOICE                     |
| bl_awb                | B/L- AWB                         | B/L- AWB                   | B/L- AWB                     | B/L- AWB                     | B/L- AWB                     | B/L- AWB                    | B/L- AWB                    |
| dam                   | DAM                              | DAM                        | DAM                          | DAM                          | DAM                          | DAM                         | DAM                         |
| booking               | —                                | BOOKING                    | BOOKING                      | BOOKING                      | BOOKING                      | BOOKING                     | BOOKING                     |
| cntr_nr               | —                                | CNTR NR                    | CNTR NR                      | CNTR NR                      | CNTR NR                      | CNTR NR                     | CNTR NR                     |
| seal                  | —                                | —                          | —                            | —                            | —                            | SEAL                        | SEAL                        |
| campania              | —                                | CAMPAÑA                    | CAMPAÑA                      | CAMPAÑA                      | SEASON                       | SEASON                      | SEASON                      |
| status                | ESTADO                           | STATUS                     | STATUS                       | STATUS                       | STATUS                       | STATUS                      | STATUS                      |
| importador            | Importador                       | —                          | —                            | —                            | —                            | IMPORTER                    | IMPORTER                    |
| buyer                 | BUYER                            | BUYER                      | BUYER                        | BUYER                        | BUYER                        | BUYER                       | BUYER                       |
| supplier              | Proveedor                        | SUPPLIER                   | SUPPLIER                     | SUPPLIER                     | SUPPLIER                     | SUPPLIER                    | SUPPLIER                    |
| proyecto              | —                                | PROYECT                    | PROYECT                      | PROYECT                      | PROYECT                      | PROYECT                     | PROYECT                     |
| categoria             | TIPO PROD                        | CATEGORY                   | CATEGORY                     | CATEGORIA                    | CATEGORY                     | CATEGORY                    | CATEGORY                    |
| producto              | PRODUCTO                         | PRODUCTO                   | PRODUCTO                     | PRODUCTO                     | PRODUCT                      | PRODUCT                     | PRODUCT                     |
| qty                   | CANT                             | QTY                        | QTY                          | QTY                          | QTY                          | QTY                         | QTY                         |
| um                    | UN                               | UM                         | UM                           | UM                           | UM                           | UM                          | UM                          |
| moneda                | —                                | CUR                        | CUR                          | CUR                          | CUR                          | CUR                         | CUR                         |
| amount                | —                                | AMOUNT                     | AMOUNT                       | AMOUNT                       | AMOUNT                       | AMOUNT                      | AMOUNT                      |
| amount_usd            | Valor Final USD                  | —                          | AMOUNT USD                   | AMOUNT USD                   | AMOUNT USD                   | AMOUNT USD                  | AMOUNT USD                  |
| payment_term          | Forma de Pago                    | PAYMENT TERM               | PAYMENT TERM                 | PAYMENT TERM                 | PAYMENT TERM                 | PAYMENT TERM                | PAYMENT TERM                |
| incoterm              | INCOTERM                         | INCOTERM                   | INCOTERM                     | INCOTERM                     | INCOTERM                     | INCOTERM                    | INCOTERM                    |
| mode                  | MODE                             | MODE                       | MODE                         | MODE                         | MODE                         | MODE                        | MODE                        |
| type                  | TYPE                             | TYPE                       | TYPE                         | TYPE                         | TYPE                         | TYPE                        | TYPE                        |
| country_origin        | —                                | —                          | COUNTRY ORIGIN               | COUNTRY ORIGIN               | COUNTRY ORIGIN               | COUNTRY ORIGIN              | COUNTRY ORIGIN              |
| pol                   | POL                              | POL                        | POL                          | POL                          | POL                          | POL                         | POL                         |
| pod                   | POD                              | POD                        | POD                          | POD                          | POD                          | POD                         | POD                         |
| ctnr_qty              | CTNR QTY                         | QTY CTNR                   | CTNR QTY                     | CTNR QTY                     | CTNR QTY                     | CTNR QTY                    | CTNR QTY                    |
| ctnr_type             | CTNR TYPE                        | TYPE CTNR                  | CTNR TYPE                    | CTNR TYPE                    | CTNR TYPE                    | CTNR TYPE                   | CTNR TYPE                   |
| bulks                 | PQ                               | BULKS                      | BULKS                        | BULKS                        | BULKS                        | BULKS                       | BULKS                       |
| bulks_type            | —                                | —                          | —                            | —                            | —                            | BULKS TYPE                  | BULKS TYPE                  |
| peso_bruto            | —                                | —                          | —                            | —                            | —                            | PESO BRUTO                  | PESO BRUTO                  |
| ffw                   | AGENTE DE CARGA                  | FFW                        | FFW                          | FFW                          | FFW                          | FFW                         | FFW                         |
| ffw_reference         | —                                | —                          | —                            | —                            | —                            | FFW REFERENCE               | FFW REFERENCE               |
| shipping_line         | NAVIERA / AEROLINEA              | NAVIERA / AEROLINEA        | NAVIERA / AEROLINEA          | NAVIERA / AEROLINEA          | SHIPPING LINE                | SHIPPING LINE               | SHIPPING LINE               |
| vessel                | NAVE                             | NAVE                       | NAVE                         | NAVE                         | VESSEL                       | VESSEL                      | VESSEL                      |
| depot                 | DEPOT                            | DEPOT                      | DEPOT                        | DEPOT                        | DEPOT                        | DEPOT                       | DEPOT                       |
| customs_agent         | AG. ADUANA                       | AG. ADUANA                 | AG. ADUANA                   | AG. ADUANA                   | CUSTOMS AGENT                | CUSTOMS AGENT               | CUSTOMS AGENT               |
| canal                 | Canal                            | CANAL                      | —                            | —                            | —                            | —                           | —                           |
| insurance_hper        | SEGURO HPER                      | INSURANCE HPER             | INSURANCE HPER               | INSURANCE HPER               | INSURANCE HPER               | INSURANCE HPER              | INSURANCE HPER              |
| delivery_type         | —                                | —                          | —                            | —                            | —                            | DELIVERY TYPE               | DELIVERY TYPE               |
| punto_llegada         | Lugar Entrega                    | PUNTO DE LLEGADA           | PUNTO DE LLEGADA             | PUNTO DE LLEGADA             | PUNTO DE LLEGADA             | FINAL DELIVERY POINT        | FINAL DELIVERY POINT        |
| transp_t1             | Transporte                       | TRANSP LIMA (T1)           | TRANSP LIMA (T1)             | TRANSP LIMA (T1)             | TRANSP LIMA (T1)             | TRANSP LIMA (T1)            | TRANSP LIMA (T1)            |
| transp_t2             | —                                | TRANSP TRUX (T2)           | TRANSP TRUX (T2)             | TRANSP TRUX (T2)             | TRANSP TRUX (T2)             | TRANSP TRUX (T2)            | TRANSP TRUX (T2)            |
| cargo_ready           | —                                | CARGO READY                | CARGO READY                  | CARGO READY                  | —                            | —                           | —                           |
| quote_request_date    | —                                | —                          | QUOTE REQUEST RATE           | QUOTE REQUEST DATE           | QUOTE REQUEST DATE           | QUOTE REQUEST DATE          | QUOTE REQUEST DATE          |
| assignment_date       | —                                | ASSIGMENT DATE             | ASSIGMENT DATE2              | ASSIGMENT DATE               | ASSIGMENT DATE               | ASSIGMENT DATE              | ASSIGMENT DATE              |
| pick_up_date          | —                                | PICK UP DATE               | PICK UP DATE                 | PICK UP DATE                 | PICK UP DATE                 | —                           | —                           |
| atd                   | ETD                              | ATD                        | ATD                          | ATD                          | ATD                          | ATD                         | ATD                         |
| ata                   | ETA                              | ATA                        | ATA                          | ATA                          | ATA                          | ATA                         | ATA                         |
| week_ata              | WEEK                             | WEEK ATA                   | WEEK ATA                     | WEEK ATA                     | WEEK ATA                     | WEEK ATA                    | WEEK ATA                    |
| ie_date               | —                                | FECHA IE                   | FECHA IE                     | FECHA IE                     | IE DATE                      | IE DATE                     | IE DATE                     |
| fecha_num_dam         | Fecha Numeración                 | FECHA NUM DAM              | FECHA NUM DAM                | FECHA NUM DAM                | FECHA NUM DAM                | FECHA NUM DAM               | FECHA NUM DAM               |
| fecha_descarga        | —                                | FECHA DESCARGA             | FECHA DESCARGA               | FECHA DESCARGA               | DOWNLOAD DATE                | —                           | —                           |
| fecha_tarja           | —                                | FECHA TARJA                | FECHA TARJA                  | FECHA TARJA                  | TALLY DATE                   | —                           | —                           |
| fecha_levante         | Fecha Levante                    | FECHA LEVANTE              | FECHA LEVANTE                | FECHA LEVANTE                | FECHA LEVANTE                | CUSTOMS RELEASE DATE        | CUSTOMS RELEASE DATE        |
| senasa_inspeccion     | —                                | FECHA INSPECCION SENASA    | FECHA INSPECCION SENASA      | FECHA INSPECCION SENASA      | FECHA INSPECCION SENASA      | SENASA INSPECTION DATE      | SENASA INSPECTION DATE      |
| senasa_liberacion     | —                                | FECHA LIBERACION SENASA    | FECHA LIBERACION SENASA      | FECHA LIBERACION SENASA      | FECHA LIBERACION SENASA      | SENASA RELEASE DATE         | SENASA RELEASE DATE         |
| demurrage_exp         | —                                | FECHA VCTO SOBRESTADIA     | FECHA VCTO SOBRESTADIA       | FECHA VCTO SOBRESTADIA       | FECHA VCTO SOBRESTADIA       | DEMURRAGE EXPIRATION DATE   | DEMURRAGE EXPIRATION DATE   |
| fecha_retiro_t1       | Fecha de Retiro y Entrega Agersa | FECHA RETIRO (T1)          | FECHA RETIRO (T1)            | FECHA RETIRO (T1)            | FECHA RETIRO (T1)            | WITHDRAWL DATE              | WITHDRAWL DATE              |
| fecha_entrega_base_t2 | —                                | FECHA ENTREGA BASE T2 (T1) | FECHA ENTREGA BASE T2 (T1)   | FECHA ENTREGA BASE T2 (T1)   | FECHA ENTREGA BASE T2 (T1)   | DELIVERY DATE BASE T2 (T1)  | DELIVERY DATE BASE T2 (T1)  |
| fecha_fin_viaje       | Fecha Entrega Trux               | FECHA FIN DE VIAJE (T2)    | FECHA FIN DE VIAJE (T2)      | FECHA FIN DE VIAJE (T2)      | FECHA FIN DE VIAJE (T2)      | FINAL ARRIVAL DATE          | FINAL ARRIVAL DATE          |
| receipt_confirmation  | —                                | —                          | FECHA CONFIRMACION RECEPCION | FECHA CONFIRMACION RECEPCION | FECHA CONFIRMACION RECEPCION | RECEIPT CONFIRMATION DATE   | RECEIPT CONFIRMATION DATE   |
| fecha_liquidacion     | Fecha Liquidación                | FECHA ENVIO ARCHIV         | FECHA ENVIO ARCHIV           | —                            | —                            | FECHA ENVIO LIQUIDACIÓN     | FECHA ENVIO LIQUIDACIÓN     |
| fecha_registro_gastos | —                                | —                          | —                            | —                            | —                            | FECHA DE REGISTRO DE GASTOS | FECHA DE REGISTRO DE GASTOS |

## Resumen de cobertura

| esquema   |   columnas_canónicas |   columnas_huérfanas |
|:----------|---------------------:|---------------------:|
| 2019-2020 |                   40 |                   33 |
| 2021      |                   55 |                   20 |
| 2022      |                   58 |                   18 |
| 2023      |                   58 |                   14 |
| 2024      |                   57 |                   12 |
| 2025      |                   62 |                   10 |
| 2026      |                   62 |                    9 |

## Columnas huérfanas por esquema (no mapeadas a canónica)


**2019-2020** (14 útiles, excluye 'Unnamed'): `USD`, `EUR`, `ASEGURADORA`, `POLIZA PROFORMA  NRO`, `Días Libres Sobrestadía`, `Dias libres Almacenaje`, `DIRECCIONAMIENTO`, `Orden`, `Solicitud Proforma`, `Solicitud Pago`, `Pago Proforma`, `Envío de Provisión`, `Fecha Envío Archivo`, `Observaciones`

**2021** (20 útiles, excluye 'Unnamed'): `STACKING`, `ETD 1`, `ETD 2`, `WEEK ATD`, `ETA 1`, `ETA 2`, `INSURANCE POLICY`, `INSURANCE POLICY DATE`, `Orden`, `GR T1 `, `FECHA GATE IN (T1)`, `FECHA INICIO DE VIAJE  (T2)`, `WEEK TRUX `, `GR T2 `, `GR HORTIFRUT`, `NRO CAJA ARCHIVO`, `COURIER PARA ENVÍO DE DOCS`, `NRO DE TRACKING DOCS `, `OBSERVACIONES HORTIFRUT`, `CARGO PRODUCE OBSERVACIONES`

**2022** (18 útiles, excluye 'Unnamed'): `CP OPENNING`, `STACKING`, `WEEK ATD`, `INSURANCE POLICY DATE`, `Orden`, `FECHA VISTO BUENO`, `GR T1 `, `FECHA GATE IN (T1)`, `FECHA INICIO DE VIAJE  (T2)`, `WEEK TRUX `, `GR T2 `, `GR HORTIFRUT`, `WEEK NECESIDAD`, `NRO CAJA ARCHIVO`, `COURIER PARA ENVÍO DE DOCS`, `NRO DE TRACKING DOCS `, `OBSERVACIONES HORTIFRUT`, `CARGO PRODUCE OBSERVACIONES`

**2023** (12 útiles, excluye 'Unnamed'): `CP OPENNING`, `WEEK ATD`, `INSURANCE POLICY DATE`, `Orden`, `GR T1 `, `WEEK BASE LIMA`, `FECHA INICIO DE VIAJE  (T2)`, `WEEK TRUX`, `GR T2 `, `GR HORTIFRUT`, `WEEK NECESIDAD`, `CP OBSERVACIONES`

**2024** (10 útiles, excluye 'Unnamed'): `CP OPENNING`, `WEEK ATD`, `INSURANCE POLICY DATE`, `ORDER`, `GR T1 `, `WEEK BASE LIMA`, `WEEK TRUX`, `GR T2 `, `GR HORTIFRUT`, `CP OBSERVACIONES`

**2025** (10 útiles, excluye 'Unnamed'): `CP OPENNING`, `WEEK ATD`, `INSURANCE POLICY DATE`, `ORDER`, `GR T1 `, `WEEK BASE T2`, `ARRIVAL WEEK`, `GR T2`, `GR HORTIFRUT`, `CP OBSERVATIONS`

**2026** (9 útiles, excluye 'Unnamed'): `WEEK ATD`, `INSURANCE POLICY DATE`, `ORDER`, `GR T1 `, `WEEK BASE T2`, `ARRIVAL WEEK`, `GR T2`, `GR HORTIFRUT`, `CP OBSERVATIONS`