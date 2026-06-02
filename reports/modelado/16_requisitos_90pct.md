# Requisitos para alcanzar una predicción del 90% (error ≤10%)

> Documento de requerimientos para llevar a negocio (Hortifrut). Define **exactamente**
> qué datos y qué arquitectura se necesitan para que la predicción de costos de importación
> pase del techo actual (**MdAPE ~25% / WAPE ~0.35**, ver `14_modelado.md §11`) a un error
> ≤10% ("90%"). Basado en la anatomía real del costo calculada sobre las 1,058 operaciones
> fiables.

---

## 0. Aclaración: "90%" en regresión

Esto es **regresión**, no clasificación: no existe "accuracy 90%". El equivalente honesto es
**MdAPE ≤ 10%** (error mediano por operación) y/o **WAPE ≤ 0.10** (error agregado). Hoy estamos
en ~25%. La pregunta correcta no es "tunear más el modelo" sino **"qué datos faltan"**.

---

## 1. Anatomía real del costo (n = 1,058 operaciones fiables)

El error vive en pocos bloques grandes. La mayor parte del costo **no es aleatoria: es
determinística** (fórmulas y tarifas). Hoy el ML *predice* cosas que en realidad son
*cálculos y lookups*.

### Peso de cada concepto en el COSTO TOTAL

| % total | concepto | naturaleza | ¿predecible hoy? |
|---:|---|---|---|
| **65.0%** | DERECHOS / tributos | **calculable exacto** (Ad Valorem × CIF) | hoy fuera del target |
| 12.6% | FLETE internacional | cotizable (quote al booking) | sí, mal |
| 5.8% | DESCARGA | tarifa por contenedor | tarifable |
| 5.7% | TRANSPORTE T2 (puerto→fundo) | distancia × tarifa | sí, mal |
| 3.2% | SOBRESTADÍA / almacenaje | **estocástico** (demoras) | no fiable |
| 2.3% | SERVICIOS LOGÍSTICOS | extraordinarios | no fiable |
| 1.4% | TRANSPORTE T1 (Callao→Lima) | tarifa | tarifable |
| 1.2% | HANDLING / THC puerto | tarifa naviera | tarifable |
| 1.1% | AGENCIAMIENTO ADUANA | tarifa fija por agente | tarifable |
| 1.0% | INSPECCIÓN / VERIFICACIÓN | tasa | tarifable |
| <1% | SENASA, SEGUROS, OTROS | tasas fijas | tarifable |

### Peso dentro de SERVICIOS (target actual = total − tributos)

| % servicios | concepto |
|---:|---|
| 36.0% | FLETE internacional |
| 16.6% | DESCARGA |
| 16.4% | TRANSPORTE T2 (fundo) |
| 9.0% | SOBRESTADÍA / almacenaje |
| 6.5% | SERVICIOS LOGÍSTICOS |
| 4.1% | TRANSPORTE T1 |
| 3.5% | HANDLING / THC |
| 3.2% | AGENCIAMIENTO ADUANA |
| 2.9% | INSPECCIÓN |
| <1% | FITOSANITARIOS, SEGUROS, OTROS |

### Disponibilidad actual de los drivers físicos (fiable)

| driver | no-nulo hoy | problema |
|---|---:|---|
| `amount_usd` (valor mercadería) | 100.0% | OK — driver #1 actual |
| `ctnr_qty` (nº contenedores) | 53.5% | la mitad falta |
| `pais_origen` | 51.9% | la mitad falta |
| **`peso_bruto`** | **9.1%** | casi todo falta — driver clave aéreo/LCL |

---

## 2. Qué nivel de "90%" es alcanzable

| objetivo | ¿alcanzable? | por qué |
|---|---|---|
| **90% en costo TOTAL** | **Sí, claramente** | 65% (tributos) + 12.6% (flete) = **78% se vuelve casi-exacto** con HS + quote |
| **90% en SERVICIOS** (target hoy) | Sí, pero más duro | flete (36%) es la única gran palanca calculable; el resto necesita tarifarios + peso + distancia |
| **90% puntual en CADA operación** | **No al 100%** | sobrestadía (9% serv) depende de demoras/inspecciones aleatorias → banda, no punto |

**Conclusión:** "90% en el agregado y en la mayoría de operaciones" **sí**; "90% exacto en cada
operación volátil" **no** — eso se gestiona con intervalo P10–P90.

---

## 3. Requisitos de datos — completo, por palanca

### A. Tributos exactos → elimina el 65% del error del total

| dato | formato | fuente |
|---|---|---|
| Partida arancelaria **HS** (10 dígitos) por producto | código | SUNAT / declaración DAM |
| Tasa **Ad Valorem** por HS | % | arancel SUNAT |
| Base **CIF** | = `amount_usd` + flete + seguro | casi disponible |
| IGV 16% + IPM 2% + percepción | reglas fijas | normativa SUNAT |

→ tributos = **fórmula determinística**, no modelo. Hoy están fuera del target de servicios;
incorporarlos vuelve trivial la predicción del **costo total**.

### B. Flete cotizado → 36% de servicios de "predicho" a "conocido"

| dato | formato |
|---|---|
| **Tarifa de flete cotizada** al booking | USD/contenedor (FCL) o USD/kg-chargeable (aéreo/LCL) |
| Recargos BAF / CAF / THC origen-destino | USD |

→ el `quote_request_date` ya existe en el esquema; **falta capturar el MONTO cotizado**.

### C. Drivers físicos → costo variable (descarga, T2, handling, aéreo)

| dato | hoy | desbloquea |
|---|---|---|
| **Peso bruto (kg)** | solo **9.1%** | costo aéreo/LCL por kg, manipuleo, modelo por-kg estable |
| **Volumen (CBM)** | falta | LCL, handling |
| **Nº real de contenedores + tipo** (20/40/HC/reefer) | **53.5%** | THC, descarga (16.6%), T1 |
| **Distancia / zona del fundo destino** (km Callao→fundo) | solo categórico (`punto_llegada`) | **transporte T2 (16.4% de servicios)** |

### D. Tarifarios (rate cards) → servicios fijos = lookup, no predicción

| concepto | % servicios | dato requerido |
|---|---:|---|
| Descarga | 16.6% | tarifa por contenedor del depot/puerto, por año |
| Agenciamiento aduana | 3.2% | tarifa fija por agente |
| Handling / THC | 3.5% | tarifa naviera |
| SENASA / inspección | 2.9% | tasas publicadas |

→ son **tarifas vigentes por proveedor × año**. Con la tabla se **calculan**, no se estiman.

### E. Moneda y tiempo

- **Tipo de cambio por factura** (fecha real de cada gasto) → elimina ruido FX.
- **Fechas de proceso completas** (numeración, levante, retiro, demurrage) → permite estimar
  la sobrestadía como `días_demora × tarifa_demurrage`.

### F. Calidad y volumen de datos

- Completar nulos históricos (peso, contenedores, país de origen).
- Más historia reciente.
- **Reentreno trimestral** (la deriva de tarifas es fuerte: el costo salta ×2.5 año a año).

---

## 4. Arquitectura objetivo: modelo HÍBRIDO (cálculo + ML)

No es un ML más grande. Es **descomponer** el costo y atacar cada bloque con la herramienta
correcta:

```
TOTAL = tributos      (fórmula HS × CIF)            # 65% — exacto
      + flete         (quote del booking)           # 12.6% — conocido
      + fijos         (rate card lookup)            # ~15% — tarifado
      + variable_ML   (peso, volumen, distancia,    # ~10% — predicho, ahora con
                       nº contenedores)             #         drivers físicos reales
      + sobrestadía   (intervalo P90, no punto)     # ~3% — estocástico → banda
```

Cada bloque grande se vuelve casi-exacto → el error agregado cae por debajo del 10%.
El ML solo carga con la parte **verdaderamente variable** (la más pequeña).

---

## 5. Resumen priorizado (qué pedir primero a negocio)

| prioridad | dato a conseguir | habilita | impacto |
|---|---|---|---|
| **1** | Partida HS + tasa Ad Valorem | tributos exactos | **65% del total** |
| **2** | Monto de flete cotizado (quote) | flete conocido | 36% de servicios |
| **3** | Peso bruto + nº contenedores histórico | costo variable real | descarga + handling + aéreo |
| **4** | Tarifarios por proveedor × año | fijos = lookup | ~25% de servicios |
| **5** | Distancia a fundo + TC por factura | transporte T2 + ruido FX | 16% serv + bias |

---

## 6. Lo genuinamente irreducible

La **sobrestadía / demurrage** (9% de servicios) depende de demoras portuarias, canal rojo de
aforo e inspecciones — es **aleatorio**. Nunca será 90% puntual: se cubre con el intervalo
**P10–P90** (ya implementado, cobertura conformal ~75%), no con una estimación exacta.

Por eso la meta correcta es: **predicción puntual ≤10% en el agregado y en la mayoría de
operaciones; banda de incertidumbre para las volátiles.**

---

## 7. Estado actual vs objetivo

| | hoy (Fases 1–2 hechas) | con Fases 3–4 (datos nuevos) |
|---|---|---|
| MdAPE despliegue | ~25% (0.249) | objetivo ≤10% |
| WAPE | 0.348 | ≤0.10 |
| tributos (65% total) | fuera del modelo | fórmula exacta |
| flete (36% serv) | predicho (error alto) | quote conocido |
| qué falla | falta peso, quote, HS, tarifas | — |

> El modelado puro (Fases 1–2) ya está al límite con los datos actuales. **El salto al 90%
> NO es un problema de algoritmo — es un problema de datos.** Depende de que negocio/operaciones
> entreguen lo de la sección 3. Sin ello, ~25% es el techo físico.

---

### Referencias

- Anatomía del costo: calculada de `data_csv/processed/target_por_concepto.parquet` (n=1,058).
- Resultados de modelado actuales: `reports/modelado/14_modelado.md` (§11) y `_modelado_summary.json`.
- Plan de mejora general: `reports/modelado/15_plan_mejora.md` (este documento detalla su Fase 3–4).
