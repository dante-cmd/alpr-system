# Acerca de ALPR System

## ¿Qué es?

**ALPR System** es un sistema de reconocimiento automático de placas vehiculares (ALPR/ANPR) orientado a producción, diseñado para identificar placas del bloque **Mercosur** (Brasil, Argentina, Uruguay, Paraguay) en imágenes, videos y streams de cámara en tiempo real.

El proyecto nace como una solución **CPU-first**: no requiere GPU para inferencia, lo que la hace accesible para despliegues en edge, servidores cloud modestos o estaciones de peaje sin aceleración NVIDIA.

## Motivación

Los sistemas comerciales de ALPR suelen ser costosos, cerrados o dependientes de hardware especializado. ALPR System busca demostrar que es posible construir un pipeline abierto, modular y eficiente usando modelos de visión por computadora de última generación, con un stack 100 % Python y despliegue sencillo mediante Docker.

## Arquitectura

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Fuente de      │     │  Pipeline ALPR   │     │  API REST + UI  │
│  video/imagen   │────▶│  (CPU)           │────▶│  web            │
│  Webcam/RTSP    │     │                  │     │                 │
└─────────────────┘     │ 1. Detección     │     │  /detect        │
                        │    de vehículo   │     │  /batch/detect  │
                        │ 2. Detección     │     │  /stream/start  │
                        │    de placa      │     │  /stream/frame  │
                        │ 3. OCR           │     │  /metrics       │
                        │ 4. Post-proceso  │     │                 │
                        └──────────────────┘     └─────────────────┘
```

## Stack tecnológico

| Capa | Tecnología |
|------|------------|
| Detección de objetos | YOLOv8 (Ultralytics) |
| OCR | PaddleOCR PP-OCRv3/v4 |
| API web | FastAPI |
| UI | HTML5 + Tailwind CSS + JavaScript vanilla |
| Métricas | Prometheus |
| Logs | structlog |
| Contenerización | Docker + Docker Compose |
| Tests | pytest |

## Características principales

- **Detección de placas** en imágenes estáticas y video.
- **Validación sintáctica** de formatos Mercosur (`ABC1D23`, `AB123CD`, `ABCD123`, `ABC1234`).
- **API REST** con endpoints de salud, detección, batch, streaming y métricas.
- **Streaming en vivo** desde webcam local, cámaras IP (RTSP) o archivos de video.
- **Interfaz web moderna** con modo oscuro, drag & drop y visor de stream.
- **Optimizaciones CPU**: skip de frames, ROI configurable, deduplicación temporal y captura en hilo independiente.
- **Evaluación y métricas** de producción: Exact Match, Character Accuracy Rate, Levenshtein, IoU, precision/recall.

## Alcance y limitaciones actuales

- El sistema funciona **sin GPU**, pero la latencia por frame en CPU es de ~300-800 ms según resolución. Para throughput alto se recomienda GPU o batch de frames.
- El detector de placas base fue entrenado con datasets internacionales genéricos. Para alcanzar ≥95 % de Exact Match en placas reales de Mercosur se requiere **fine-tuning** con datos regionales (por ejemplo, UFPR-ALPR o RodoSol-ALPR).
- No incluye reconocimiento de matrículas de motocicletas pequeñas ni placas no Mercosur sin reentrenamiento.

## Roadmap

- [x] Pipeline end-to-end (detección + OCR + post-proceso)
- [x] API REST con métricas Prometheus
- [x] Streaming de video en tiempo real
- [x] Interfaz web moderna
- [ ] Fine-tuning con datasets Mercosur
- [ ] Exportación a ONNX / OpenVINO para inferencia más rápida en CPU
- [ ] Soporte de múltiples cámaras simultáneas
- [ ] WebSocket para streaming de frames sin polling
- [ ] Autenticación y roles de usuario

## Evidencia y referencias

La elección de YOLOv8 y PaddleOCR se basa en resultados publicados en la literatura reciente de ALPR (Satya et al. 2025, Laroca et al. IJCNN 2018/IET ITS 2021) y en el reporte técnico de PaddleOCR 3.0.

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Ver [`LICENSE`](LICENSE).

## Contacto / Contribuciones

Las contribuciones son bienvenidas. Para reportar bugs o proponer mejoras, abre un issue en el repositorio.
