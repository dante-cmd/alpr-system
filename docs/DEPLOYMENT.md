# Guía de despliegue

## Requisitos de hardware

- **CPU**: x86_64 con 4+ cores; AVX2 recomendado.
- **RAM**: 4 GB mínimo para inferencia CPU; 8 GB recomendado.
- **GPU**: opcional; NVIDIA con CUDA 11.8+ para entrenamiento o TensorRT.
- **Disco**: 5 GB libres para modelos, dependencias y logs.

## Variables de entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `ALPR_API_HOST` | Host de la API | `0.0.0.0` |
| `ALPR_API_PORT` | Puerto de la API | `8000` |
| `ALPR_CONF_THRESHOLD` | Confianza mínima detección | `0.5` |
| `ALPR_DETECT_VEHICLE_FIRST` | Detectar vehículo antes que placa | `true` |
| `ALPR_JSON_LOGS` | Logs en JSON | `false` |
| `ALPR_ENABLE_PROMETHEUS` | Métricas Prometheus | `true` |

## Despliegue con Docker

```bash
docker compose -f alpr/docker/docker-compose.yml up --build -d
```

Con observabilidad:

```bash
docker compose -f alpr/docker/docker-compose.yml --profile observability up --build -d
```

## Rollback

```bash
docker compose -f alpr/docker/docker-compose.yml down
docker pull alpr-system:<version-anterior>
docker compose -f alpr/docker/docker-compose.yml up -d
```

## Monitoreo

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Métricas de la API: http://localhost:8000/metrics

## Interfaz web

La API incluye una UI moderna servida en la raíz (`/`). No requiere compilación ni Node.

```bash
python -m alpr.api.main
# Abrir http://localhost:8000
```

Funcionalidades disponibles en el navegador:
- **Imagen**: drag & drop de una foto, previsualización y resultados con bounding boxes.
- **Batch**: subir múltiples imágenes y ver tabla de resultados.
- **Stream en vivo**: iniciar/detener webcam, RTSP o archivo de video; ver frame anotado y placas detectadas en tiempo real.
- **Configuración**: resumen de variables de entorno.
- **Logs**: consola de eventos del cliente.

## Instalación manual en CPU

Para evitar incompatibilidades entre `paddleocr>=3.x` y `paddlepaddle` en entornos CPU-only, se recomienda instalar las dependencias en este orden:

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch CPU (debe coincidir con torchvision)
pip install torch==2.12.0+cpu torchvision==0.27.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# PaddleOCR estable (evitar 3.x mientras arrastra paddlex y modelos PP-OCRv6 con OneDNN)
pip install paddlepaddle>=2.6.0,<4.0.0 paddleocr>=2.9.0,<3.0.0

# Resto del proyecto
pip install -e ".[dev]"
```

Si ya existe una instalación rota con `paddleocr>=3.x`, eliminar el entorno virtual y reinstalar desde cero suele ser más rápido que limpiar manualmente los artefactos de `paddlex`.

## Troubleshooting

### `ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`

Esta excepción proviene del runtime OneDNN de PaddlePaddle cuando PaddleOCR 3.x descarga modelos `PP-OCRv6_*`. Las opciones probadas son:

1. **(Recomendada)** Usar `paddleocr>=2.9.0,<3.0.0`, que descarga modelos `PP-OCRv3`/`PP-OCRv4` estables. Es la configuración actual del repositorio.
2. Forzar `FLAGS_use_mkldnn=False` y/o `FLAGS_pir_apply_pass=0` antes de importar `paddle`.
3. Descargar manualmente los infer de PP-OCRv4 y apuntar `det_model_dir`/`rec_model_dir` en `PaddleOCR(...)`.

### `operator torchvision::nms does not exist`

Ocurre cuando `torchvision` no coincide con `torch` o se instala la variante CUDA en CPU. Reinstalar ambos desde el índice `cpu`:

```bash
pip install torch==2.12.0+cpu torchvision==0.27.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

## Streaming de video

La API expone endpoints para procesar streams de webcam, cámaras IP (RTSP) o archivos de video.

### Endpoints de stream

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/stream/start` | Inicia captura y procesamiento desde una fuente |
| `POST` | `/stream/stop` | Detiene el stream |
| `GET`  | `/stream/status` | Estado del stream activo |
| `GET`  | `/stream/frame` | Último frame anotado (JPEG) |
| `GET`  | `/stream/plates` | Placas detectadas recientemente |

### Ejemplo: webcam local

```bash
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": 0, "process_every_n_frames": 3, "max_fps": 2}'

# Ver frame anotado
curl http://localhost:8000/stream/frame -o frame.jpg

# Ver placas detectadas
curl http://localhost:8000/stream/plates

# Detener
curl -X POST http://localhost:8000/stream/stop
```

### Ejemplo: cámara IP Hikvision por RTSP

```bash
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "source": "rtsp://usuario:pass@192.168.1.100:554/Streaming/Channels/101",
    "process_every_n_frames": 3,
    "max_fps": 2,
    "roi": [200, 300, 1400, 900],
    "detect_vehicle": true
  }'
```

### Parámetros de `/stream/start`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `source` | requerido | Índice de webcam (`0`), URL RTSP o ruta a archivo |
| `process_every_n_frames` | `3` | Procesa 1 de cada N frames |
| `max_fps` | `5.0` | Máximo de FPS a procesar |
| `roi` | `null` | Región de interés `[x1, y1, x2, y2]` para reducir zona de búsqueda |
| `min_confidence` | `0.5` | Confianza mínima OCR para reportar placa |
| `detect_vehicle` | `true` | `false` si la imagen ya es una placa recortada |

### Eficiencia en streaming

- El hilo de captura es independiente para no perder frames RTSP.
- Se procesa solo 1 de cada N frames y se respeta `max_fps`.
- Usa ROI para limitar el área de detección.
- Deduplicación temporal: una misma placa no se reporta repetidamente durante `dedup_ttl_seconds` (default 5 s).
- En CPU-only se recomienda `max_fps=1..2` y resolución de entrada ≤1080p.

## Recomendación de cámaras

Para este proyecto CPU-only lo más importante es una imagen nítida, buena iluminación y poca compresión.

### Opción recomendada: cámara IP con RTSP

**Hikvision DS-2CD2047G2-LU** (o similar de 4 MP):
- Resolución 4 MP (2688×1520), suficiente para leer placas a 5-15 m.
- WDR 120 dB y EXIR 2.0 para contraluz y baja luz.
- Soporte RTSP H.264/H.265, PoE, lente 2.8 mm (angular) o 4 mm.
- Precio aproximado: USD 120-180.

Para instalaciones donde la placa ocupe pocos píxeles, usar la versión con lente motorizado o fija de 6-8 mm para acercar el campo de visión.

### Alternativa profesional: cámara ANPR dedicada

Si el presupuesto lo permite, una cámara con firmware ANPR mejora mucho la tasa de lectura:
- **Hikvision iDS-2CD7A26G0/P-IZHSY**
- **Dahua ITC237-PW1A-IRZ**

Estas cámaras entregan ya la placa por RTSP/HTTP, pero también puedes usar su stream de video y correr el pipeline propio para mantener flexibilidad de formato (Mercosur) y métricas.

### Consejos de configuración

1. **Resolución**: configura 1920×1080 o 2560×1440. Más allá de 4 MP no mejora el OCR en CPU y aumenta la latencia.
2. **Bitrate/Códec**: usa H.264 con bitrate 4-6 Mbps. H.265 reduce ancho de banda pero aumenta carga de decodificación CPU.
3. **Iluminación**: si es nocturno, usa cámara con IR o iluminación blanca auxiliar. Evita reflejos en el parabrisas.
4. **Ángulo**: instala la cámara lo más frontal posible (< 30° horizontal) y a la altura de la placa.
5. **ROI**: configura la región de interés en `/stream/start` para no procesar toda la imagen.

## CI/CD

El workflow `.github/workflows/ci.yml` ejecuta lint, tests y build de Docker en cada push a `main`/`develop`.
