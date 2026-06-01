# Entregable 4 — Features candidatas finales (conocidas a la llegada)


## Mercadería / valor

| feature         | tratamiento                                         | nota                  |
|:----------------|:----------------------------------------------------|:----------------------|
| amount_usd      | log1p + imputar 0→mediana                           | numérica, sesgada     |
| qty             | log1p; revisar unidades (UM)                        | numérica              |
| categoria_canon | one-hot (8 grupos)                                  | categórica baja card. |
| producto        | descartar como feature (alta card., usar categoría) | alta card.            |

## Transporte / modo

| feature    | tratamiento                                     | nota                          |
|:-----------|:------------------------------------------------|:------------------------------|
| mode       | one-hot (AIR/SEA/LAND)                          | driver primario               |
| type       | one-hot (FCL/LCL/AIR)                           | categórica                    |
| es_aereo   | binaria derivada                                | flag                          |
| ctnr_qty   | imputar 0 en aéreo; numérica                    | condicional marítimo          |
| bulks      | log1p; imputar mediana                          | driver de handling/transporte |
| peso_bruto | usar si se completa histórico; hoy solo 2025-26 | condicional                   |

## Ruta / origen

| feature     | tratamiento                            | nota        |
|:------------|:---------------------------------------|:------------|
| pol         | frequency/target encoding (86 cat.)    | alta card.  |
| pod         | one-hot (28, mayoría Callao/Lima)      | media card. |
| pais_origen | imputar desde POL; one-hot top + OTROS | 53% nulo    |
| zona_ruta   | derivar POL→POD agrupado               | derivada    |

## Comercial / contractual

| feature        | tratamiento                     | nota                       |
|:---------------|:--------------------------------|:---------------------------|
| incoterm       | one-hot (10)                    | alta importancia           |
| incoterm_grupo | one-hot (4) derivada            | driver de quién paga flete |
| payment_term   | numérica; imputar mediana       | numérica                   |
| supplier       | target encoding suavizado (174) | alta card.                 |

## Proveedores logísticos

| feature       | tratamiento               | nota                    |
|:--------------|:--------------------------|:------------------------|
| shipping_line | target/freq encoding (66) | driver fuerte (ε²=0.47) |
| ffw           | frequency encoding (39)   | media                   |
| customs_agent | one-hot (15)              | media                   |
| depot         | frequency encoding (28)   | driver fuerte (ε²=0.46) |

## Temporales / tiempos

| feature         | tratamiento                             | nota                |
|:----------------|:----------------------------------------|:--------------------|
| transit_days    | ata−atd; imputar mediana por modo       | derivada            |
| mes_arribo      | cíclica (sin/cos) o one-hot             | estacionalidad      |
| anio            | ordinal; clave para validación temporal | tendencia/inflación |
| canal_rojo      | binaria; OJO sólo tras numeración DAM   | condicional         |
| requiere_senasa | binaria derivada                        | plantas/sustratos   |
| tiene_seguro    | binaria (insurance_hper)                | flag                |