import { PolarController } from './controller.js';

export function drawChart(canvas, points, { now = Date.now(), windowMs = 10000, unit = 'µV', gapMs = 1000 } = {}) {
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
  const ratio = globalThis.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const visible = points.filter(p => p.t >= now - windowMs && p.t <= now);
  const values = visible.flatMap(p => p.values);
  const lo = 0;
  let hi = values.length ? Math.max(...values) : 1;
  const margin = Math.max((hi - lo) * 0.1, 1);
  hi += margin;
  ctx.font = '12px sans-serif';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(`${hi.toFixed(0)} ${unit}`, 8, 16);
  ctx.fillText(`${lo.toFixed(0)} ${unit}`, 8, h - 25);
  ctx.fillText(`−${windowMs / 1000}s`, 65, h - 6);
  ctx.fillText('now', w - 40, h - 6);
  if (!visible.length) ctx.fillText('Waiting for current samples', 100, h / 2);
  {
    const color = '#38bdf8', channel = 0;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    let prev;
    for (const p of visible) {
      const x = 65 + ((p.t - (now - windowMs)) / windowMs) * (w - 80);
      const y = 22 + (1 - (p.values[channel] - lo) / (hi - lo)) * (h - 55);
      if (!prev || p.t - prev.t > gapMs) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      prev = p;
    }
    ctx.stroke();
    if (visible.length === 1) {
      const p = visible[0];
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(65 + ((p.t - (now - windowMs)) / windowMs) * (w - 80),
        22 + (1 - (p.values[channel] - lo) / (hi - lo)) * (h - 55), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

export function mountPolarPanel(root) {
  root.classList.add('panel', 'detail-card');
  root.style.marginBottom = '18px';
  root.innerHTML = `
    <style>
      #polar-panel canvas { height: 180px; }
      #polar-panel button { background: #1f2937; color: #e5e7eb; border: 1px solid #64748b;
        border-radius: 8px; padding: 8px 12px; cursor: pointer; }
      #polar-panel button:disabled { opacity: .5; cursor: default; }
      #polar-panel button:focus-visible { outline: 2px solid #38bdf8; outline-offset: 2px; }
      @media (max-width: 700px) {
        #polar-panel .detail-head { flex-wrap: wrap; gap: 8px; }
      }
    </style>
    <div class="detail-head"><h2>EXG · 200 ms RMS</h2>
      <div><button type="button" id="polar-connect">Connect Polar H10</button>
      <button type="button" id="polar-disconnect" disabled>Disconnect</button></div></div>
    <p id="polar-status" role="status" aria-live="polite"></p>
    <p><strong id="polar-rms-value">—</strong> µV RMS</p>
    <canvas id="polar-rms" width="1200" height="180" aria-label="Live EXG 200 millisecond RMS in microvolts"></canvas>`;
  const el = id => root.querySelector(`#polar-${id}`);
  const text = (id, value) => {
    const node = el(id);
    if (node.textContent !== value) node.textContent = value;
  };
  const controller = new PolarController({ onChange: render });
  function render() {
    const c = controller;
    const connected = c.state === 'connected';
    text('status', connected
      ? `${c.device.name || 'Polar H10'} · ${c.data.fresh() ? 'Receiving EXG' : 'EXG missing or stale'}`
      : c.message);
    el('connect').disabled = c.busy || c.wanted;
    el('disconnect').disabled = !c.wanted && !c.busy;
    const latest = c.data.rms.at(-1);
    text('rms-value', connected && c.data.fresh() && latest ? latest.values[0].toFixed(1) : '—');
    drawChart(el('rms'), c.data.rms);
  }
  el('connect').addEventListener('click', () => void controller.connect());
  el('disconnect').addEventListener('click', () => controller.disconnect());
  let timer = setInterval(render, 100);
  const pause = () => { clearInterval(timer); controller.disconnect(); };
  const resume = (event) => {
    if (event.persisted) {
      clearInterval(timer);
      timer = setInterval(render, 100);
      render();
    }
  };
  const dispose = () => {
    pause();
    window.removeEventListener('pagehide', pause);
    window.removeEventListener('pageshow', resume);
  };
  window.addEventListener('pagehide', pause);
  window.addEventListener('pageshow', resume);
  render();
  return { controller, dispose };
}

const root = document.getElementById('polar-panel');
if (root) mountPolarPanel(root);
