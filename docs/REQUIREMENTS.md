# Fase 1: Requerimientos y Dataset

## Requerimientos funcionales

- Detectar placas en imágenes y video.
- Reconocer caracteres alfanuméricos de placas Mercosur.
- Validar formato sintáctico de la placa.
- Exponer API REST con endpoints de health, detección, batch y métricas.
- Soportar despliegue con Docker y Docker Compose.

## Requerimientos no funcionales

| Parámetro | Objetivo |
|-----------|----------|
| Latencia p95 CPU | ≤ 300 ms por imagen 720p |
| Throughput CPU | ≥ 5 img/s |
| Resolución mínima | 640×480 |
| Exact Match | ≥ 95 % (día) |
| Recall detección | ≥ 98 % |
| Hardware objetivo | CPU x86_64; GPU NVIDIA opcional |

## Supuestos

- Las imágenes de entrada tienen suficiente resolución para leer la placa a ojo humano.
- Las placas siguen formatos Mercosur (Brasil, Argentina, Uruguay, Paraguay).
- El modelo de detector de placas base requiere fine-tuning regional para métricas de producción.

## Riesgos

- Falta de GPU en entorno de desarrollo actual impide entrenar y validar métricas reales.
- Datasets públicos de calidad (UFPR-ALPR, RodoSol-ALPR) requieren acuerdo académico.
- Modelos base genéricos pueden no alcanzar Exact Match ≥ 95 % en Mercosur sin fine-tuning.

## Datasets públicos identificados

```yaml
dataset: UFPR-ALPR
país: Brasil
n_imágenes: 4,500
licencia: Académica (solicitud por email institucional)
fortalezas: Escenarios reales, cámara en movimiento, anotaciones detalladas
 debilidades: Tamaño reducido, formato antiguo de placas brasileñas
uso_recomendado: Entrenamiento y validación de detección/OCR regional
```

```yaml
dataset: RodoSol-ALPR
país: Brasil / Mercosur
n_imágenes: 20,000
licencia: Académica (solicitud por email institucional)
fortalezas: Gran volumen, placas Mercosur y antiguas, 4 esquinas anotadas, peajes reales
 debilidades: Requiere acuerdo; nocturno limitado
uso_recomendado: Entrenamiento principal y test de producción
```

```yaml
dataset: CCPD
país: China
n_imágenes: >250,000
licencia: Académica
fortalezas: Masivo, condiciones variadas
 debilidades: Chino; no aplica directamente a Mercosur
uso_recomendado: Data augmentation / pretraining de detector genérico
```

## Estrategia de splits

- Train: 40 % (RodoSol) + UFPR-ALPR según protocolo autor.
- Validation: 20 % (RodoSol).
- Test in-distribution: 40 % (RodoSol) + UFPR-ALPR test.
- Test out-of-distribution: imágenes propias nocturnas, lluvia, ángulo, oclusión.
