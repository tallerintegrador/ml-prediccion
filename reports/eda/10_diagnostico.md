# 4.7 — Diagnóstico para el modelado


## Adecuación muestral

- Operaciones con target fiable: **1,058**.
- Total joineadas (incl. parciales): 1,234; despachadas sin facturar: 196 (inferencia futura, sin target).
- Features candidatas crudas: ~24; tras encoding la dimensionalidad sube (one-hot de incoterm/pol/categoría).
- **Riesgo de sobreajuste moderado**: 1,058 filas es suficiente para árboles con regularización (LightGBM), pero NO para one-hot masivo de alta cardinalidad. Usar target/frequency encoding y CV temporal. El gasto se concentra en pocos segmentos (SUSTRATOS×SEA=46%), así que conviene reportar error por segmento.

## Riesgos de LEAKAGE (excluir como features)

Variables/columnas conocidas SÓLO después de la facturación o la liquidación:
| elemento                                                      | motivo                                                                        |
|:--------------------------------------------------------------|:------------------------------------------------------------------------------|
| Conceptos de costo desglosados                                | Monto Final, Igv, Importe Total, Monto Total USD del expense — SON el target. |
| fecha_liquidacion / fecha_registro_gastos                     | fechas contables posteriores a la facturación.                                |
| Nro Liquid., Nro. de Doc., Tipo de Doc.                       | identificadores de la factura logística (no existen al predecir).             |
| status = LIQUIDADO/LIQUIPARCIAL                               | el estado de liquidación se conoce DESPUÉS; no usar como feature.             |
| fecha_levante / SENASA / receipt_confirmation                 | parcialmente posteriores a la llegada; usar con cautela o sólo su previsión.  |
| igv_usd, tributos_usd (como features del modelo de servicios) | se calculan de la factura; modelar aparte, no como input.                     |

## Recomendaciones de modelado

- **Target:** modelar `log1p(servicios)` (skew≈9 en crudo) o usar objetivo Gamma/Tweedie; back-transform con corrección de sesgo (smearing). Tributos: modelo aparte o cálculo arancel×CIF.
- **Encoding:** one-hot para baja card. (mode, type, incoterm, categoría, pod); target/frequency encoding suavizado para alta card. (supplier, pol, shipping_line, depot, ffw).
- **Nulos:** numéricas → mediana (por modo cuando aplique); categóricas → 'DESCONOCIDO'; contenedor → 0 en aéreo (no es faltante real).
- **Outliers:** no eliminar (operaciones caras legítimas); robustez vía log y modelos basados en árboles; winsorizar sólo errores claros de captura.
- **Validación:** **forward chaining por campaña** (entrenar ≤año t, validar año t+1); NUNCA K-fold aleatorio (hay deriva temporal +15%/año).
- **Métricas:** MAE y WAPE (negocio: provisión total), MdAPE (robusta al sesgo), RMSE (penaliza grandes), pinball loss + cobertura de intervalos (para la pre-liquidación con incertidumbre).

## Datos adicionales que convendría recopilar

- **Peso bruto histórico** (hoy sólo 2025-2026): clave para costo aéreo y unitario.
- **Partida arancelaria (HS) / tasa Ad Valorem**: haría los tributos calculables.
- **Tipo de cambio por factura** y fecha: para multimoneda exacta.
- **Tarifa de flete cotizada** (quote) al momento del embarque.
- **Distancia/zona fundo destino** y nº real de contenedores por operación.