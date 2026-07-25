const state = {
  view: "live",
  cameras: [],
  modules: [],
  events: [],
  system: null,
  config: null,
  models: null,
  cameraDraft: {
    name: "",
    camera_id: "",
    rtsp_url: "",
    device: "",
    stream_role: "both",
    detect_fps: "0",
    modules: [],
  },
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

function isEditingForm() {
  const el = document.activeElement;
  return Boolean(el && el.closest && el.closest("#view form"));
}

function syncCameraDraftFromForm(form) {
  if (!form) return;
  const fd = new FormData(form);
  state.cameraDraft = {
    name: String(fd.get("name") || ""),
    camera_id: String(fd.get("camera_id") || ""),
    rtsp_url: String(fd.get("rtsp_url") || ""),
    device: String(fd.get("device") || ""),
    stream_role: String(fd.get("stream_role") || "both"),
    detect_fps: String(fd.get("detect_fps") ?? "0"),
    modules: [...form.querySelectorAll('input[name="modules"]:checked')].map((el) => el.value),
  };
}

function resetCameraDraft() {
  state.cameraDraft = {
    name: "",
    camera_id: "",
    rtsp_url: "",
    device: "",
    stream_role: "both",
    detect_fps: "0",
    modules: [],
  };
}

function setView(view) {
  // Persist draft before leaving Cameras (in case form will be destroyed).
  syncCameraDraftFromForm(document.getElementById("add-camera-form"));
  state.view = view;
  document.getElementById("view-title").textContent = viewTitles[view] || view;
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  render({ force: true });
}

function updateChrome() {
  const cameras = state.cameras || [];
  const online = cameras.filter((c) => c.runtime?.online).length;
  document.getElementById("pill-cams").textContent = `${online}/${cameras.length} online`;
  document.getElementById("pill-device").textContent =
    state.system?.device || state.config?.hardware?.device || "—";
  const cpu = state.system?.cpu_percent ?? 0;
  const ram = state.system?.ram_percent ?? 0;
  document.getElementById("sys-mini").textContent = `CPU ${cpu}% · RAM ${ram}%`;
}

function cameraRowsHtml() {
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
  return rows || `<tr><td colspan="6" class="muted">No cameras</td></tr>`;
}

function patchCamerasTable() {
  const tbody = document.querySelector("#cameras-table tbody");
  if (!tbody) return false;
  tbody.innerHTML = cameraRowsHtml();
  bindCameraTableHandlers();
  return true;
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
    updateChrome();

    // Never rebuild the Add Camera form while it exists / while typing.
    if (state.view === "cameras" && document.getElementById("add-camera-form")) {
      syncCameraDraftFromForm(document.getElementById("add-camera-form"));
      patchCamerasTable();
      return;
    }
    // Soft-update Live tiles so MJPEG keeps streaming while FPS refreshes.
    if (state.view === "live" && document.querySelector("[data-cam-card]")) {
      if (patchLiveCards()) return;
    }
    if (isEditingForm()) {
      return;
    }
    render();
  } catch (error) {
    // Do not destroy an in-progress form on transient API errors.
    if (document.getElementById("add-camera-form") || isEditingForm()) {
      console.warn("API refresh failed:", error);
      return;
    }
    document.getElementById("view").innerHTML =
      `<div class="card" style="padding:16px;color:#ffb4b4">API error: ${esc(error.message)}</div>`;
  }
}

function formatFps(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "0.0";
  return n.toFixed(1);
}

function liveFpsLabel(cam) {
  const cap = formatFps(cam.runtime?.capture_fps);
  const det = formatFps(cam.runtime?.detect_fps);
  return `FPS ${cap}<span class="muted-fps">det ${det}</span>`;
}

function patchLiveCards() {
  const enabled = state.cameras.filter((c) => c.enabled);
  const cards = [...document.querySelectorAll("[data-cam-card]")];
  if (!cards.length) return false;
  const names = new Set(enabled.map((c) => c.name));
  // If camera set changed, do a full re-render.
  if (cards.length !== enabled.length || cards.some((el) => !names.has(el.dataset.camCard))) {
    return false;
  }
  enabled.forEach((cam) => {
    const card = document.querySelector(`[data-cam-card="${CSS.escape(cam.name)}"]`);
    if (!card) return;
    const online = Boolean(cam.runtime?.online);
    const badge = card.querySelector("[data-live-badge]");
    if (badge) {
      badge.className = `badge ${online ? "on" : "off"}`;
      badge.textContent = online ? "LIVE" : "OFF";
    }
    const fps = card.querySelector("[data-live-fps]");
    if (fps) fps.innerHTML = liveFpsLabel(cam);
    const metaFps = card.querySelector("[data-meta-fps]");
    if (metaFps) {
      metaFps.textContent = `Current FPS ${formatFps(cam.runtime?.capture_fps)} · detect ${formatFps(
        cam.runtime?.detect_fps
      )}`;
    }
    const device = card.querySelector("[data-meta-device]");
    if (device) device.textContent = `device ${cam.runtime?.device || cam.device || "global"}`;
  });
  return true;
}

function renderLive() {
  if (!state.cameras.length) {
    return `<div class="card" style="padding:20px"><div class="muted">No cameras yet. Add one under Cameras.</div></div>`;
  }
  return `<div class="grid cams" id="live-grid">${state.cameras
    .filter((c) => c.enabled)
    .map((cam) => {
      const online = cam.runtime?.online;
      const src = `/api/streams/${encodeURIComponent(cam.name)}/mjpeg`;
      return `<article class="card" data-cam-card="${esc(cam.name)}">
        <div class="card-h">
          <div>
            <h3>${esc(cam.name)}</h3>
            <div class="muted">${esc((cam.modules || []).join(", ") || "no modules")}</div>
          </div>
          <span class="badge ${online ? "on" : "off"}" data-live-badge>${online ? "LIVE" : "OFF"}</span>
        </div>
        <div class="feed-wrap">
          <div class="fps-pill" data-live-fps>${liveFpsLabel(cam)}</div>
          <img class="feed" src="${src}" alt="${esc(cam.name)}" />
        </div>
        <div class="meta">
          <div data-meta-fps>Current FPS ${formatFps(cam.runtime?.capture_fps)} · detect ${formatFps(
            cam.runtime?.detect_fps
          )}</div>
          <div data-meta-device>device ${esc(cam.runtime?.device || cam.device || "global")}</div>
          <div><a href="/player.html?src=${encodeURIComponent(cam.name)}" target="_blank" rel="noreferrer">Open WebRTC</a></div>
        </div>
      </article>`;
    })
    .join("")}</div>`;
}

function renderCameras() {
  const d = state.cameraDraft;
  const moduleChecks = state.modules
    .map((m) => {
      const checked = d.modules.includes(m.id) ? "checked" : "";
      return `<label class="check"><input type="checkbox" name="modules" value="${esc(m.id)}" ${checked} ${
        m.enabled ? "" : "disabled"
      } /> ${esc(m.id)}</label>`;
    })
    .join("");

  const deviceOpts = [
    ["", "Global default"],
    ["cuda:0", "GPU (cuda:0)"],
    ["cpu", "CPU"],
  ]
    .map(
      ([value, label]) =>
        `<option value="${esc(value)}" ${d.device === value ? "selected" : ""}>${esc(label)}</option>`
    )
    .join("");

  const roleOpts = [
    ["both", "Detect + Live"],
    ["detect", "Detect only"],
    ["live", "Live only"],
  ]
    .map(
      ([value, label]) =>
        `<option value="${esc(value)}" ${d.stream_role === value ? "selected" : ""}>${esc(label)}</option>`
    )
    .join("");

  return `
  <div class="grid" style="grid-template-columns: 1.1fr 1fr; align-items:start">
    <form class="form" id="add-camera-form" autocomplete="off">
      <h3 style="margin:0">Add Camera</h3>
      <div class="form-row two">
        <div class="form-row"><label>Name</label><input name="name" required placeholder="gate-1" value="${esc(
          d.name
        )}" /></div>
        <div class="form-row"><label>Camera ID</label><input name="camera_id" placeholder="optional external id" value="${esc(
          d.camera_id
        )}" /></div>
      </div>
      <div class="form-row"><label>RTSP URL</label><input name="rtsp_url" required placeholder="rtsp://user:pass@host:554/stream" value="${esc(
        d.rtsp_url
      )}" /></div>
      <div class="form-row two">
        <div class="form-row">
          <label>Hardware</label>
          <select name="device">${deviceOpts}</select>
        </div>
        <div class="form-row">
          <label>Stream role</label>
          <select name="stream_role">${roleOpts}</select>
        </div>
      </div>
      <div class="form-row"><label>Detect FPS (0 = global)</label><input name="detect_fps" type="number" step="0.1" value="${esc(
        d.detect_fps
      )}" /></div>
      <div class="form-row"><label>Detection modules</label><div class="checks">${moduleChecks}</div></div>
      <button class="btn" type="submit">Add Camera</button>
    </form>
    <div class="card" style="padding:0; overflow:auto">
      <table id="cameras-table">
        <thead><tr><th>Camera</th><th>RTSP</th><th>Modules</th><th>HW</th><th>State</th><th></th></tr></thead>
        <tbody>${cameraRowsHtml()}</tbody>
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

function render(options = {}) {
  const root = document.getElementById("view");
  // Keep live form values if the form is about to be rebuilt (nav switch).
  syncCameraDraftFromForm(document.getElementById("add-camera-form"));

  // Soft path: cameras form already mounted and this is a background refresh.
  if (!options.force && state.view === "cameras" && document.getElementById("add-camera-form")) {
    patchCamerasTable();
    return;
  }
  if (!options.force && state.view === "live" && document.querySelector("[data-cam-card]")) {
    if (patchLiveCards()) return;
  }

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

function bindCameraTableHandlers() {
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
}

function bindViewHandlers() {
  const addForm = document.getElementById("add-camera-form");
  if (addForm) {
    const persist = () => syncCameraDraftFromForm(addForm);
    addForm.addEventListener("input", persist);
    addForm.addEventListener("change", persist);

    addForm.onsubmit = async (event) => {
      event.preventDefault();
      syncCameraDraftFromForm(addForm);
      const d = state.cameraDraft;
      const payload = {
        name: d.name,
        camera_id: d.camera_id || "",
        rtsp_url: d.rtsp_url,
        device: d.device || "",
        stream_role: d.stream_role || "both",
        detect_fps: Number(d.detect_fps || 0),
        modules: d.modules,
        enabled: true,
      };
      await api("/api/cameras", { method: "POST", body: JSON.stringify(payload) });
      resetCameraDraft();
      await refresh();
      setView("live");
    };
  }

  bindCameraTableHandlers();

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
