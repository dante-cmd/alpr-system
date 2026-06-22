const API = '';
const logEl = document.getElementById('logs-container');

function log(level, message) {
  const line = document.createElement('div');
  const t = new Date().toLocaleTimeString();
  const colors = { info: 'text-gray-300', warn: 'text-yellow-400', error: 'text-red-400', success: 'text-green-400' };
  line.className = `${colors[level] || 'text-gray-300'} whitespace-pre-wrap`;
  line.textContent = `[${t}] [${level.toUpperCase()}] ${message}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

document.getElementById('clear-logs').addEventListener('click', () => logEl.innerHTML = '');

// Health check
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const badge = document.getElementById('health-badge');
    if (res.ok) {
      badge.innerHTML = '<span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span><span>Online</span>';
      badge.className = 'px-3 py-1.5 rounded-full text-xs font-medium bg-green-500/10 text-green-400 flex items-center gap-2';
    } else throw new Error('not ok');
  } catch (e) {
    const badge = document.getElementById('health-badge');
    badge.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span><span>Offline</span>';
    badge.className = 'px-3 py-1.5 rounded-full text-xs font-medium bg-red-500/10 text-red-400 flex items-center gap-2';
  }
}
setInterval(checkHealth, 3000);
checkHealth();

// Tabs
const tabs = document.querySelectorAll('.tab-btn');
const contents = document.querySelectorAll('.tab-content');
tabs.forEach(btn => {
  btn.addEventListener('click', () => {
    tabs.forEach(t => { t.classList.remove('tab-active'); t.classList.add('text-gray-400'); });
    contents.forEach(c => c.classList.add('hidden'));
    btn.classList.add('tab-active');
    btn.classList.remove('text-gray-400');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove('hidden');
  });
});

// Image tab
const dropZone = document.getElementById('drop-zone');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const imagePreviewContainer = document.getElementById('image-preview-container');
const detectBtn = document.getElementById('detect-btn');
const imageResult = document.getElementById('image-result');

let selectedImage = null;

dropZone.addEventListener('click', () => imageInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('border-primary', 'bg-gray-900/50'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-primary', 'bg-gray-900/50'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('border-primary', 'bg-gray-900/50');
  if (e.dataTransfer.files.length) handleImage(e.dataTransfer.files[0]);
});
imageInput.addEventListener('change', e => { if (e.target.files.length) handleImage(e.target.files[0]); });

function handleImage(file) {
  selectedImage = file;
  const url = URL.createObjectURL(file);
  imagePreview.src = url;
  imagePreviewContainer.classList.remove('hidden');
  detectBtn.disabled = false;
  imageResult.innerHTML = '<p class="text-gray-400">Listo para procesar.</p>';
}

detectBtn.addEventListener('click', async () => {
  if (!selectedImage) return;
  detectBtn.disabled = true;
  detectBtn.innerHTML = '<span class="loader inline-block align-middle mr-2"></span>Procesando...';
  imageResult.innerHTML = '';
  const form = new FormData();
  form.append('file', selectedImage);
  try {
    const res = await fetch(`${API}/detect`, { method: 'POST', body: form });
    const data = await res.json();
    renderImageResult(data);
    log('success', `Imagen procesada: ${data.count} placa(s)`);
  } catch (e) {
    imageResult.innerHTML = `<div class="text-red-400">Error: ${e.message}</div>`;
    log('error', e.message);
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detectar placas';
  }
});

function renderImageResult(data) {
  if (!data.plates || data.plates.length === 0) {
    imageResult.innerHTML = '<p class="text-gray-400">No se detectaron placas.</p>';
    return;
  }
  let html = `<div class="mb-2 text-xs text-gray-400">Tiempo: ${data.processing_time_ms} ms</div>`;
  html += '<div class="space-y-3">';
  data.plates.forEach(p => {
    html += `
      <div class="plate-card bg-gray-900 rounded-xl p-4 border border-gray-800">
        <div class="flex items-center justify-between">
          <div class="text-2xl font-bold tracking-wider text-white">${escapeHtml(p.plate)}</div>
          <span class="px-2 py-1 rounded text-xs font-medium ${p.valid_format ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'}">${p.valid_format ? 'Válido' : 'Inválido'}</span>
        </div>
        <div class="grid grid-cols-2 gap-2 mt-3 text-xs text-gray-400">
          <div>Conf. OCR: <span class="text-gray-200">${(p.confidence * 100).toFixed(1)}%</span></div>
          <div>Conf. detección: <span class="text-gray-200">${(p.detection_confidence * 100).toFixed(1)}%</span></div>
          <div class="col-span-2">Bbox: ${JSON.stringify(p.bbox)}</div>
        </div>
      </div>`;
  });
  html += '</div>';
  imageResult.innerHTML = html;
}

// Batch tab
const batchInput = document.getElementById('batch-input');
const batchBtn = document.getElementById('batch-btn');
const batchResults = document.getElementById('batch-results');

batchInput.addEventListener('change', () => batchBtn.disabled = batchInput.files.length === 0);

batchBtn.addEventListener('click', async () => {
  const files = batchInput.files;
  if (!files.length) return;
  batchBtn.disabled = true;
  batchBtn.innerHTML = '<span class="loader inline-block align-middle mr-2"></span>Procesando...';
  const form = new FormData();
  for (const f of files) form.append('files', f);
  try {
    const res = await fetch(`${API}/batch/detect`, { method: 'POST', body: form });
    const data = await res.json();
    renderBatchResults(data.results);
    log('success', `Batch completado: ${files.length} archivo(s)`);
  } catch (e) {
    batchResults.innerHTML = `<div class="text-red-400">Error: ${e.message}</div>`;
    log('error', e.message);
  } finally {
    batchBtn.disabled = false;
    batchBtn.textContent = 'Procesar imágenes';
  }
});

function renderBatchResults(results) {
  if (!results || !results.length) {
    batchResults.innerHTML = '<p class="text-gray-400">Sin resultados.</p>';
    return;
  }
  let html = '<table class="w-full text-sm text-left"><thead><tr class="text-gray-400 border-b border-gray-800"><th class="py-2">Archivo</th><th>Placas</th><th>Tiempo (ms)</th></tr></thead><tbody>';
  results.forEach(r => {
    const plates = r.error ? `<span class="text-red-400">${escapeHtml(r.error)}</span>` : (r.result.plates || []).map(p => `<span class="inline-block bg-gray-800 px-2 py-0.5 rounded mr-1">${escapeHtml(p.plate)}</span>`).join('');
    const time = r.error ? '-' : r.result.processing_time_ms;
    html += `<tr class="border-b border-gray-800/50"><td class="py-2">${escapeHtml(r.filename)}</td><td>${plates}</td><td>${time}</td></tr>`;
  });
  html += '</tbody></table>';
  batchResults.innerHTML = html;
}

// Stream tab
const streamSourceType = document.getElementById('stream-source-type');
const streamSourceValue = document.getElementById('stream-source-value');
const streamStartBtn = document.getElementById('stream-start-btn');
const streamStopBtn = document.getElementById('stream-stop-btn');
const streamFrame = document.getElementById('stream-frame');
const streamPlaceholder = document.getElementById('stream-placeholder');
const streamPlates = document.getElementById('stream-plates');
const streamStatus = document.getElementById('stream-status');
const streamFpsBadge = document.getElementById('stream-fps-badge');
const streamProcessingBadge = document.getElementById('stream-processing-badge');

let streamInterval = null;
let platesInterval = null;
let frameTimestamps = [];

streamSourceType.addEventListener('change', () => {
  const map = { webcam: '0', rtsp: 'rtsp://usuario:pass@192.168.1.100:554/Streaming/Channels/101', file: 'data/samples/video.mp4' };
  streamSourceValue.value = map[streamSourceType.value];
});

streamStartBtn.addEventListener('click', async () => {
  const source = streamSourceValue.value;
  const skip = parseInt(document.getElementById('stream-skip').value, 10) || 3;
  const fps = parseFloat(document.getElementById('stream-fps').value) || 2;
  const roiRaw = document.getElementById('stream-roi').value.trim();
  const roi = roiRaw ? roiRaw.split(',').map(Number) : null;
  const detectVehicle = document.getElementById('stream-detect-vehicle').checked;
  const payload = { source, process_every_n_frames: skip, max_fps: fps, detect_vehicle: detectVehicle };
  if (roi && roi.length === 4 && roi.every(n => !isNaN(n))) payload.roi = roi;

  try {
    streamStartBtn.disabled = true;
    const res = await fetch(`${API}/stream/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Error iniciando stream');
    streamStatus.innerHTML = `<span class="text-green-400">●</span> Stream activo: ${data.source}`;
    startStreamPolling();
    log('success', `Stream iniciado: ${data.source}`);
  } catch (e) {
    streamStatus.innerHTML = `<span class="text-red-400">●</span> Error: ${e.message}`;
    log('error', e.message);
  } finally {
    streamStartBtn.disabled = false;
  }
});

streamStopBtn.addEventListener('click', async () => {
  try {
    const res = await fetch(`${API}/stream/stop`, { method: 'POST' });
    await res.json();
    stopStreamPolling();
    streamStatus.innerHTML = '<span class="text-gray-500">●</span> Stream detenido';
    streamFrame.classList.add('hidden');
    streamPlaceholder.classList.remove('hidden');
    streamPlates.innerHTML = '<div class="text-sm text-gray-500">Sin placas detectadas aún.</div>';
    streamFpsBadge.textContent = '-- FPS';
    log('info', 'Stream detenido');
  } catch (e) {
    log('error', e.message);
  }
});

function startStreamPolling() {
  stopStreamPolling();
  streamPlaceholder.classList.add('hidden');
  streamFrame.classList.remove('hidden');
  streamProcessingBadge.classList.remove('hidden');

  const loadFrame = async () => {
    try {
      const res = await fetch(`${API}/stream/frame?t=${Date.now()}`);
      if (res.ok) {
        const blob = await res.blob();
        streamFrame.src = URL.createObjectURL(blob);
        const now = performance.now();
        frameTimestamps.push(now);
        frameTimestamps = frameTimestamps.filter(t => now - t < 1000);
        streamFpsBadge.textContent = `${frameTimestamps.length} FPS`;
      }
    } catch (e) {
      // Silencioso: puede fallar entre frames
    }
  };

  const loadPlates = async () => {
    try {
      const res = await fetch(`${API}/stream/plates?limit=20`);
      if (res.ok) {
        const data = await res.json();
        renderStreamPlates(data.plates);
      }
    } catch (e) {
      // Silencioso
    }
  };

  loadFrame();
  loadPlates();
  streamInterval = setInterval(loadFrame, 800);
  platesInterval = setInterval(loadPlates, 1500);
}

function stopStreamPolling() {
  if (streamInterval) clearInterval(streamInterval);
  if (platesInterval) clearInterval(platesInterval);
  streamInterval = null;
  platesInterval = null;
  streamProcessingBadge.classList.add('hidden');
}

function renderStreamPlates(plates) {
  if (!plates || !plates.length) {
    streamPlates.innerHTML = '<div class="text-sm text-gray-500">Sin placas detectadas aún.</div>';
    return;
  }
  streamPlates.innerHTML = plates.slice().reverse().map(p => `
    <div class="plate-card bg-gray-900 rounded-lg p-3 border border-gray-800 flex items-center justify-between">
      <div>
        <div class="text-lg font-bold tracking-wider">${escapeHtml(p.plate)}</div>
        <div class="text-xs text-gray-500">OCR ${(p.confidence * 100).toFixed(1)}% · Det ${(p.detection_confidence * 100).toFixed(1)}%</div>
      </div>
      <span class="px-2 py-1 rounded text-xs font-medium ${p.valid_format ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'}">${p.valid_format ? 'Válido' : 'Inválido'}</span>
    </div>
  `).join('');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

log('info', 'Interfaz cargada. Esperando conexión con la API...');
