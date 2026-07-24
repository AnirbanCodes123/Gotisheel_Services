const state = {
  view: "live",
  cameras: [],
  modules: [],
  events: [],
  system: null,
  config: null,
  models: null,
};

const viewTitles = {
  live: "Live",
  cameras: "Cameras",
  events: "Events",
  modules: "Modules",
  models: "Models",
  system: "System",
  settings: "Settings",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  if (response.status === 204) return null;
  const ct = response.headers.get("content-type") || "";
  return ct.includes("application/json") ? response.json() : response.text();
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setView(view) {
  state.view = view;
  document.getElementById("view-title").textContent = viewTitles[view] || view;
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  render();
}

async function refresh() {
  try {
    const [cameras, modules, events, system, config, models] = await Promise.all([
      api("/api/cameras"),
      api("/api/modules"),
      api("/api/events?limit=40"),
      api("/api/system"),
      api("/api/config"),
      api("/api/models"),
    ]);
    state.cameras = cameras;
    state.modules = modules;
    state.events = events;
    state.system = system;
    state.config = config;
    state.models = models;
    const online = cameras.filter((c) => c.runtime?.online).length;
    document.getElementById("pill-cams").textContent = `${online}/${cameras.length} online`;
    document.getElementById("pill-device").textContent = system?.device || config?.hardware?.device || "—";
    const cpu = system?.cpu_percent ?? 0;
    const ram = system?.ram_percent ?? 0;
    document.getElementById("sys-mini").textContent = `CPU ${cpu}% · RAM ${ram}%`;
    render();
  } catch (error) {
    document.getElementById("view").innerHTML = `<div class="card" style="padding:16px;color:#ffb4b4">API error: ${esc(error.message)}</div>`;
  }
}

function renderLive() {
  if (!state.cameras.length) {
    return `<div class="card" style="padding:20px"><div class="muted">No cameras yet. Add one under Cameras.</div></div>`;
  }
  return `<div class="grid cams">${state.cameras
    .filter((c) => c.enabled)
    .map((cam) => {
      const online = cam.runtime?.online;
      const src = `/api/streams/${encodeURIComponent(cam.name)}/mjpeg`;
      return `<article class="card">
        <div class="card-h">
          <div>
            <h3>${esc(cam.name)}</h3>
            <div class="muted">${esc((cam.modules || []).join(", ") || "no modules")}</div>
          </div>
          <span class="badge ${online ? "on" : "off"}">${online ? "LIVE" : "OFF"}</span>
        </div>
        <img class="feed" src="${src}" alt="${esc(cam.name)}" />
        <div class="meta">
          <div>cap ${esc(cam.runtime?.capture_fps ?? 0)} · detect ${esc(cam.runtime?.detect_fps ?? 0)} FPS</div>
          <div>device ${esc(cam.runtime?.device || cam.device || "global")}</div>
          <div><a href="${esc(cam.webrtc_url)}" target="_blank" rel="noreferrer">Open WebRTC</a></div>
        </div>
      </article>`;
    })
    .join("")}</div>`;
}

function renderCameras() {
  const moduleChecks = state.modules
    .map(
      (m) =>
        `<label class="check"><input type="checkbox" name="modules" value="${esc(m.id)}" ${m.enabled ? "" : "disabled"} /> ${esc(m.id)}</label>`
    )
    .join("");
  const rows = state.cameras
    .map((cam) => {
      return `<tr>
        <td><strong>${esc(cam.name)}</strong><div class="muted">${esc(cam.camera_id)}</div></td>
        <td class="muted" style="max-width:240px;overflow:hidden;text-overflow:ellipsis">${esc(cam.rtsp_url)}</td>
        <td>${esc((cam.modules || []).join(", ") || "—")}</td>
        <td>${esc(cam.device || "global")}</td>
        <td><span class="badge ${cam.enabled ? "on" : "off"}">${cam.enabled ? "on" : "off"}</span></td>
        <td>
          <button class="btn ghost" data-toggle="${cam.id}">${cam.enabled ? "Disable" : "Enable"}</button>
          <button class="btn danger" data-del="${cam.id}">Delete</button>
        </td>
      </tr>`;
    })
    .join("");

  return `
  <div class="grid" style="grid-template-columns: 1.1fr 1fr; align-items:start">
    <form class="form" id="add-camera-form">
      <h3 style="margin:0">Add Camera</h3>
      <div class="form-row two">
        <div class="form-row"><label>Name</label><input name="name" required placeholder="gate-1" /></div>
        <div class="form-row"><label>Camera ID</label><input name="camera_id" placeholder="optional external id" /></div>
      </div>
      <div class="form-row"><label>RTSP URL</label><input name="rtsp_url" required placeholder="rtsp://user:pass@host:554/stream" /></div>
      <div class="form-row two">
        <div class="form-row">
          <label>Hardware</label>
          <select name="device">
            <option value="">Global default</option>
            <option value="cuda:0">GPU (cuda:0)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div class="form-row">
          <label>Stream role</label>
          <select name="stream_role">
            <option value="both">Detect + Live</option>
            <option value="detect">Detect only</option>
            <option value="live">Live only</option>
          </select>
        </div>
      </div>
      <div class="form-row"><label>Detect FPS (0 = global)</label><input name="detect_fps" type="number" step="0.1" value="0" /></div>
      <div class="form-row"><label>Detection modules</label><div class="checks">${moduleChecks}</div></div>
      <button class="btn" type="submit">Add Camera</button>
    </form>
    <div class="card" style="padding:0; overflow:auto">
      <table>
        <thead><tr><th>Camera</th><th>RTSP</th><th>Modules</th><th>HW</th><th>State</th><th></th></tr></thead>
        <tbody>${rows || `<tr><td colspan="6" class="muted">No cameras</td></tr>`}</tbody>
      </table>
    </div>
  </div>`;
}

function renderEvents() {
  if (!state.events.length) return `<div class="card" style="padding:16px" class="muted">No events yet</div>`;
  return `<div class="card">${state.events
    .map((ev) => {
      const thumb = ev.thumbnail_path
        ? `/api/event-media?path=${encodeURIComponent(ev.thumbnail_path)}`
        : "";
      return `<div class="event-row">
        ${thumb ? `<img src="${thumb}" alt="" onerror="this.style.visibility='hidden'" />` : `<div></div>`}
        <div>
          <strong>${esc(ev.label)}</strong> · ${esc(ev.camera_name)}
          <div class="muted">${esc(ev.created_at)} · uploaded=${esc(ev.uploaded)} · ${esc(ev.module_id)}</div>
          <div class="muted">${esc(JSON.stringify(ev.detail || {}))}</div>
        </div>
      </div>`;
    })
    .join("")}</div>`;
}

function renderModules() {
  return `<div class="grid cams">${state.modules
    .map((m) => {
      const cfg = m.config || {};
      return `<article class="card" style="padding:14px">
        <div class="card-h" style="padding:0 0 10px;border:0">
          <h3>${esc(m.id)}</h3>
          <span class="badge ${m.enabled ? "on" : "off"}">${m.enabled ? "enabled" : "off"}</span>
        </div>
        <div class="muted">labels: ${(m.labels || []).map(esc).join(", ")}</div>
        <div class="muted" style="margin-top:8px;font-family:var(--mono)">model: ${esc(m.model || "—")}</div>
        <pre style="white-space:pre-wrap;color:#9fb0c4;font-size:11px;background:#0d131b;padding:10px;border-radius:10px;overflow:auto">${esc(JSON.stringify(cfg, null, 2))}</pre>
      </article>`;
    })
    .join("")}</div>
    <p class="muted" style="margin-top:12px">Edit module parameters in <code>backend/config/default.yaml</code> then POST /api/config/reload and restart workers.</p>`;
}

function renderModels() {
  const disk = state.models?.disk || [];
  const registered = state.models?.registered || [];
  return `
  <div class="grid" style="grid-template-columns: 1fr 1fr; align-items:start; gap:16px">
    <form class="form" id="upload-model-form">
      <h3 style="margin:0">Upload YOLO .pt</h3>
      <div class="form-row"><label>File</label><input type="file" name="file" accept=".pt" required /></div>
      <div class="form-row"><label>Name</label><input name="name" placeholder="optional display name" /></div>
      <div class="form-row">
        <label>Bind module</label>
        <select name="module_id">
          <option value="">—</option>
          ${state.modules.map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("")}
        </select>
      </div>
      <button class="btn" type="submit">Upload</button>
    </form>
    <div class="card" style="padding:14px">
      <h3 style="margin-top:0">On disk</h3>
      <ul class="muted">${disk.map((d) => `<li>${esc(d.filename)} (${esc(d.size_mb)} MB)</li>`).join("") || "<li>No .pt files in data/models</li>"}</ul>
      <h3>Registered</h3>
      <ul class="muted">${registered.map((r) => `<li>${esc(r.name)} → ${esc(r.filename)} [${esc(r.module_id || "-")}]</li>`).join("") || "<li>None</li>"}</ul>
    </div>
  </div>`;
}

function renderSystem() {
  const s = state.system || {};
  const gpus = Array.isArray(s.gpus) ? s.gpus.filter((g) => g.index !== undefined) : [];
  const shards = s.shards || {};
  return `
  <div class="stat-grid">
    <div class="stat"><div class="muted">CPU</div><div class="v">${esc(s.cpu_percent ?? 0)}%</div><div class="bar"><i style="width:${Number(s.cpu_percent || 0)}%"></i></div></div>
    <div class="stat"><div class="muted">RAM</div><div class="v">${esc(s.ram_percent ?? 0)}%</div><div class="bar"><i style="width:${Number(s.ram_percent || 0)}%"></i></div></div>
    <div class="stat"><div class="muted">Cameras</div><div class="v">${esc(shards.camera_count ?? 0)}</div></div>
    <div class="stat"><div class="muted">Infer queue</div><div class="v">${esc(shards.scheduler?.queue_depth ?? 0)}</div></div>
  </div>
  <div class="card" style="margin-top:16px;padding:14px">
    <h3 style="margin-top:0">GPUs</h3>
    ${
      gpus.length
        ? gpus
            .map(
              (g) => `<div style="margin-bottom:12px">
                <div><strong>${esc(g.name)}</strong> · util ${esc(g.util_percent)}%</div>
                <div class="muted">mem ${esc(g.mem_used_mb)} / ${esc(g.mem_total_mb)} MB</div>
                <div class="bar"><i style="width:${Number(g.util_percent || 0)}%"></i></div>
              </div>`
            )
            .join("")
        : `<div class="muted">No NVIDIA metrics (pynvml unavailable or no GPU)</div>`
    }
  </div>
  <div class="card" style="margin-top:16px;padding:14px">
    <h3 style="margin-top:0">Shards / go2rtc</h3>
    <pre style="white-space:pre-wrap;color:#9fb0c4;font-size:12px">${esc(
      JSON.stringify(
        {
          logical_shards: shards.logical_shards,
          cameras_per_worker: shards.cameras_per_worker,
          scheduler: shards.scheduler,
          go2rtc: shards.go2rtc,
        },
        null,
        2
      )
    )}</pre>
  </div>`;
}

function renderSettings() {
  const wh = state.config?.webhook || {};
  const hw = state.config?.hardware || {};
  return `<div class="form">
    <h3 style="margin:0">Runtime configuration</h3>
    <div class="muted">Source: backend/config/default.yaml (and optional local.yaml). Env: GOTISHEEL_*</div>
    <div class="form-row"><label>Hardware device</label><input value="${esc(hw.device)}" disabled /></div>
    <div class="form-row"><label>FFmpeg hwaccel</label><input value="${esc(hw.ffmpeg_hwaccel)}" disabled /></div>
    <div class="form-row"><label>Webhook</label><input value="${esc(wh.server_url)}${esc(wh.upload_endpoint)}" disabled /></div>
    <div class="form-row"><label>Webhook enabled</label><input value="${esc(wh.enabled)}" disabled /></div>
    <button class="btn ghost" type="button" id="btn-cfg-reload">Reload YAML config</button>
  </div>`;
}

function render() {
  const root = document.getElementById("view");
  const map = {
    live: renderLive,
    cameras: renderCameras,
    events: renderEvents,
    modules: renderModules,
    models: renderModels,
    system: renderSystem,
    settings: renderSettings,
  };
  root.innerHTML = (map[state.view] || renderLive)();
  bindViewHandlers();
}

function bindViewHandlers() {
  const addForm = document.getElementById("add-camera-form");
  if (addForm) {
    addForm.onsubmit = async (event) => {
      event.preventDefault();
      const fd = new FormData(addForm);
      const modules = [...addForm.querySelectorAll('input[name="modules"]:checked')].map((el) => el.value);
      const payload = {
        name: fd.get("name"),
        camera_id: fd.get("camera_id") || "",
        rtsp_url: fd.get("rtsp_url"),
        device: fd.get("device") || "",
        stream_role: fd.get("stream_role") || "both",
        detect_fps: Number(fd.get("detect_fps") || 0),
        modules,
        enabled: true,
      };
      await api("/api/cameras", { method: "POST", body: JSON.stringify(payload) });
      await refresh();
      setView("live");
    };
  }

  document.querySelectorAll("[data-del]").forEach((btn) => {
    btn.onclick = async () => {
      if (!confirm("Delete camera?")) return;
      await api(`/api/cameras/${btn.dataset.del}`, { method: "DELETE" });
      await refresh();
    };
  });

  document.querySelectorAll("[data-toggle]").forEach((btn) => {
    btn.onclick = async () => {
      const cam = state.cameras.find((c) => String(c.id) === String(btn.dataset.toggle));
      if (!cam) return;
      await api(`/api/cameras/${cam.id}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled: !cam.enabled }),
      });
      await refresh();
    };
  });

  const upload = document.getElementById("upload-model-form");
  if (upload) {
    upload.onsubmit = async (event) => {
      event.preventDefault();
      const fd = new FormData(upload);
      await api("/api/models/upload", { method: "POST", body: fd });
      await refresh();
    };
  }

  const cfg = document.getElementById("btn-cfg-reload");
  if (cfg) {
    cfg.onclick = async () => {
      await api("/api/config/reload", { method: "POST" });
      await refresh();
    };
  }
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.onclick = () => setView(btn.dataset.view);
});

document.getElementById("btn-reload").onclick = async () => {
  await api("/api/cameras/reload", { method: "POST" });
  await refresh();
};

refresh();
setInterval(refresh, 4000);
