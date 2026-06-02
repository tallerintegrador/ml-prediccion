# Entregable 5 — Tabla de decisiones por variable

| variable              | rol     | dispo   | importancia   |   pct_nulos |   cardinalidad | decision   | tratamiento                                |
|:----------------------|:--------|:--------|:--------------|------------:|---------------:|:-----------|:-------------------------------------------|
| ctnr_qty              | feature | Sí      | alta          |        46.5 |             23 | USAR       | numérica: imputar mediana                  |
| campania              | feature | Sí      | alta          |        12   |              6 | USAR       | one-hot                                    |
| categoria             | feature | Sí      | alta          |         0   |             16 | USAR       | one-hot                                    |
| incoterm              | feature | Sí      | alta          |         0.1 |             10 | USAR       | one-hot                                    |
| mode                  | feature | Sí      | alta          |         0.2 |              3 | USAR       | one-hot                                    |
| country_origin        | feature | Sí      | alta          |        48.1 |             23 | USAR       | target/freq encoding                       |
| pol                   | feature | Sí      | alta          |         0.4 |             83 | USAR       | target/freq encoding                       |
| payment_term          | feature | Sí      | baja          |         4.5 |             12 | USAR       | numérica: imputar mediana                  |
| importador            | feature | Sí      | baja          |        69.8 |              4 | USAR       | one-hot                                    |
| buyer                 | feature | Sí      | baja          |         0   |             24 | USAR       | target/freq encoding                       |
| um                    | feature | Sí      | baja          |         0.5 |             19 | USAR       | one-hot                                    |
| moneda                | feature | Sí      | baja          |        12.1 |              2 | USAR       | one-hot                                    |
| insurance_hper        | feature | Sí      | baja          |         2.3 |              3 | USAR       | one-hot                                    |
| qty                   | feature | Sí      | media         |         0   |            450 | USAR       | numérica: imputar mediana                  |
| amount                | feature | Sí      | media         |         0   |            674 | USAR       | numérica: imputar mediana                  |
| amount_usd            | feature | Sí      | media         |         0   |            794 | USAR       | numérica: imputar mediana + log            |
| bulks                 | feature | Sí      | media         |         0   |            151 | USAR       | numérica: imputar mediana                  |
| status                | feature | Parcial | media         |         0   |              3 | USAR       | one-hot                                    |
| supplier              | feature | Sí      | media         |         0   |            143 | USAR       | target/freq encoding                       |
| producto              | feature | Sí      | media         |         0   |            275 | USAR       | target/freq encoding                       |
| type                  | feature | Sí      | media         |         0.1 |              7 | USAR       | one-hot                                    |
| pod                   | feature | Sí      | media         |         0.1 |             28 | USAR       | target/freq encoding                       |
| ctnr_type             | feature | Sí      | media         |        45.7 |              6 | USAR       | one-hot                                    |
| ffw                   | feature | Sí      | media         |         2.9 |             35 | USAR       | target/freq encoding                       |
| shipping_line         | feature | Sí      | media         |         2.8 |             62 | USAR       | target/freq encoding                       |
| depot                 | feature | Sí      | media         |         6.1 |             28 | USAR       | target/freq encoding                       |
| customs_agent         | feature | Sí      | media         |         0.9 |             14 | USAR       | one-hot                                    |
| canal                 | feature | Parcial | media         |        66.8 |              4 | USAR       | one-hot                                    |
| punto_llegada         | feature | Sí      | media         |         6.2 |             32 | USAR       | target/freq encoding                       |
| ata                   | feature | Sí      | alta          |         0.4 |            627 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| ie_date               | feature | Sí      | baja          |        24.7 |            438 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| senasa_inspeccion     | feature | Parcial | baja          |        66.2 |            173 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| senasa_liberacion     | feature | Parcial | baja          |        65.6 |            181 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| fecha_fin_viaje       | feature | Parcial | baja          |        17.3 |            563 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| atd                   | feature | Sí      | media         |         0.4 |            662 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| fecha_num_dam         | feature | Sí      | media         |        16.5 |            528 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| fecha_descarga        | feature | Sí      | media         |        40.5 |            393 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| fecha_levante         | feature | Parcial | media         |        26.3 |            494 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| demurrage_exp         | feature | Sí      | media         |        71.6 |            147 | DERIVAR    | extraer tiempos/flags, no usar fecha cruda |
| peso_bruto            | feature | Parcial | alta          |        90.9 |             71 | DESCARTAR  | 90.9% nulos                                |
| cargo_ready           | feature | Sí      | baja          |        81.8 |            135 | DESCARTAR  | 81.8% nulos                                |
| delivery_type         | feature | Sí      | baja          |        81.9 |              3 | DESCARTAR  | 81.9% nulos                                |
| fecha_liquidacion     | leakage | No      | baja          |        53.4 |             67 | EXCLUIR    | data leakage (posterior a la llegada)      |
| fecha_registro_gastos | leakage | No      | baja          |        97.4 |             18 | EXCLUIR    | data leakage (posterior a la llegada)      |

## Resumen

| decisión   |   n |
|:-----------|----:|
| USAR       |  29 |
| DERIVAR    |  10 |
| DESCARTAR  |   3 |
| EXCLUIR    |   2 |