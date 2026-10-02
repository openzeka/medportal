/* MedPortal · OpenZeka — SPA logic (no dependencies) */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------- Toast ---------------- */
function toast(kind, title, text) {
  const box = $("toasts");
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  const t = document.createElement("div");
  t.className = "t-title";
  t.textContent = title;
  el.appendChild(t);
  if (text) {
    const p = document.createElement("div");
    p.textContent = text;
    el.appendChild(p);
  }
  box.appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4500);
}

/* ---------------- Lightbox ---------------- */
function openImage(url) {
  const lb = $("lightbox");
  $("lightbox-img").src = url;
  lb.classList.remove("hidden");
}
$("lightbox").onclick = () => {
  $("lightbox").classList.add("hidden");
  $("lightbox-img").src = "";
};

/* ---------------- Fetch helper ---------------- */
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail || ""; } catch (e) { detail = ""; }
    throw new Error(`${r.status}${detail ? " " + detail : ""}`);
  }
  return r.json();
}

/* ---------------- Status pills ---------------- */
function setPill(id, state, txtId, txt) {
  const p = $(id);
  if (!p) return;
  p.classList.remove("pill-ok", "pill-warn", "pill-err", "pill-off");
  p.classList.add("pill-" + state);
  if (txtId) $(txtId).textContent = txt;
}

async function status() {
  try {
    const s = await fetchJSON("/api/status");
    setPill("pill-gpu", "ok", "st-gpu", s.gpu || "?");
    setPill("pill-radar", s.radar_ready ? "ok" : "warn", "st-radar", s.radar_ready ? "ready" : "loading");
    setPill("pill-cf", s.clinfusion_ready ? "ok" : "warn", "st-cf", s.clinfusion_ready ? "ready" : "loading");
    const busyTxt = s.busy
      ? (s.current_job ? "busy · " + s.current_job : "busy") + (s.queue_len > 0 ? ` (${s.queue_len})` : "")
      : "idle";
    setPill("pill-busy", s.busy ? "warn" : "ok", "st-busy", busyTxt);
    $("chat-status").textContent = s.clinfusion_ready ? "" : "32B loading…";
  } catch (e) {
    ["pill-gpu", "pill-radar", "pill-cf", "pill-busy"].forEach((id) => setPill(id, "err"));
    $("st-gpu").textContent = "–";
    $("st-radar").textContent = "–";
    $("st-cf").textContent = "–";
    $("st-busy").textContent = "no server";
  }
}
setInterval(status, 3000);
status();

/* ---------------- Tabs ---------------- */
function showTab(which) {
  $("view-radar").classList.toggle("hidden", which !== "radar");
  $("view-chat").classList.toggle("hidden", which !== "chat");
  $("tab-radar").classList.toggle("seg-active", which === "radar");
  $("tab-chat").classList.toggle("seg-active", which === "chat");
  $("tab-radar").setAttribute("aria-selected", which === "radar");
  $("tab-chat").setAttribute("aria-selected", which === "chat");
}
$("tab-radar").onclick = () => showTab("radar");
$("tab-chat").onclick = () => showTab("chat");

/* ---------------- RADAR: upload + polling ---------------- */
let radarFindings = [];
let radarUploadId = null;
let radarFileName = "";
let sortDesc = true;
let sliceUrl = null;
let radarPollToken = 0;
let radarAbort = null;
let uploading = false;

const POLL_INTERVAL = 1500;
const POLL_MAX_ATTEMPTS = 400;            // RADAR: ~10 min (backend RADAR_TIMEOUT 600 s)
const CHAT_POLL_MAX_ATTEMPTS = 1300;      // chat: ~32 min (backend CHAT_TIMEOUT 1800 s; 3D generations can be slow)

function setDropEnabled(on) {
  const d = $("drop");
  d.classList.toggle("busy", !on);
  d.setAttribute("aria-disabled", String(!on));
}

$("drop").onclick = () => { if (!uploading) $("file").click(); };
$("drop").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    if (!uploading) $("file").click();
  }
});
$("drop").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("drag"); };
$("drop").ondragleave = () => $("drop").classList.remove("drag");
$("drop").ondrop = (e) => {
  e.preventDefault();
  $("drop").classList.remove("drag");
  if (uploading) return;
  if (e.dataTransfer.files[0]) uploadRadar(e.dataTransfer.files[0]);
};
$("file").onchange = () => {
  const f = $("file").files[0];
  $("file").value = "";
  if (f) uploadRadar(f);
};

function cancelRadarPolling() {
  radarPollToken++;
  if (radarAbort) {
    radarAbort.abort();
    radarAbort = null;
  }
}

function setFindingState(state) {
  $("findings-empty").classList.toggle("hidden", state !== "empty");
  $("findings-loading").classList.toggle("hidden", state !== "loading");
  document.querySelector(".findings-table").classList.toggle("hidden", state !== "table");
}

async function uploadRadar(file) {
  if (uploading) return;
  uploading = true;
  setDropEnabled(false);
  cancelRadarPolling();
  radarFileName = file.name;
  $("file-name").textContent = file.name;
  $("findings-loading-sub").textContent = "Uploading CT…";
  setFindingState("loading");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const up = await fetchJSON("/api/upload", { method: "POST", body: fd });
    if (!up.upload_id) throw new Error("no upload_id returned");
    radarUploadId = up.upload_id;
    const n = Math.max(1, up.n_slices || 1);
    $("slice-range").max = n - 1;
    $("slice-range").value = Math.floor(n / 2);
    $("slice-num").max = n - 1;
    syncSliceNum();
    $("slice-total").textContent = "/ " + n;
    $("viewer-empty").classList.add("hidden");
    $("slice").classList.remove("hidden");
    $("viewer-hud").classList.remove("hidden");
    resetZoom();
    await loadSlice();
    $("findings-loading-sub").textContent = "Loading the model; the first case may take ~1 minute.";
    const j = await fetchJSON("/api/radar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: up.upload_id }),
    });
    pollRadar(j.job_id);
  } catch (e) {
    setFindingState("empty");
    toast("err", "Upload error", e.message);
  } finally {
    uploading = false;
    setDropEnabled(true);
  }
}

/* ---------------- Viewer: WC/WW, zoom, keyboard ---------------- */
const WIN = { wc: 0, ww: 2000 };
const ZOOM = { scale: 1, x: 0, y: 0 };
const ZOOM_STEPS = [1, 1.5, 2, 3, 4, 6, 8];

function applyTransform() {
  $("slice").style.transform = `translate(${ZOOM.x}px, ${ZOOM.y}px) scale(${ZOOM.scale})`;
  $("zoom-info").textContent = Math.round(ZOOM.scale * 100) + "%";
  $("viewer-stage").classList.toggle("zoomed", ZOOM.scale > 1);
}
function resetZoom() {
  ZOOM.scale = 1; ZOOM.x = 0; ZOOM.y = 0;
  applyTransform();
}
function setZoom(scale) {
  ZOOM.scale = Math.max(1, Math.min(8, scale));
  if (ZOOM.scale === 1) { ZOOM.x = 0; ZOOM.y = 0; }
  applyTransform();
}
$("zoom-in").onclick = () => {
  const i = ZOOM_STEPS.findIndex((s) => s > ZOOM.scale + 0.01);
  setZoom(i === -1 ? 8 : ZOOM_STEPS[i]);
};
$("zoom-out").onclick = () => {
  const lower = ZOOM_STEPS.filter((s) => s < ZOOM.scale - 0.01);
  setZoom(lower.length ? lower[lower.length - 1] : 1);
};
$("zoom-reset").onclick = resetZoom;

$("viewer-stage").addEventListener("wheel", (e) => {
  if (!radarUploadId) return;
  e.preventDefault();
  setZoom(ZOOM.scale * (e.deltaY < 0 ? 1.25 : 0.8));
}, { passive: false });

let pan = null;
$("viewer-stage").addEventListener("pointerdown", (e) => {
  if (ZOOM.scale <= 1 || !radarUploadId) return;
  pan = { sx: e.clientX, sy: e.clientY, ox: ZOOM.x, oy: ZOOM.y };
  $("viewer-stage").setPointerCapture(e.pointerId);
});
$("viewer-stage").addEventListener("pointermove", (e) => {
  if (!pan) return;
  ZOOM.x = pan.ox + (e.clientX - pan.sx);
  ZOOM.y = pan.oy + (e.clientY - pan.sy);
  applyTransform();
});
$("viewer-stage").addEventListener("pointerup", () => { pan = null; });

function syncSliceNum() { $("slice-num").value = $("slice-range").value; }

function setSlice(idx) {
  const max = Number($("slice-range").max) || 0;
  $("slice-range").value = Math.max(0, Math.min(idx, max));
  syncSliceNum();
  loadSlice();
}

async function loadSlice() {
  if (!radarUploadId) return;
  const idx = Number($("slice-range").value) || 0;
  $("slice-num").value = idx;
  $("viewer-hud").textContent = `WL ${WIN.wc} / WW ${WIN.ww} · slice ${idx}`;
  try {
    const r = await fetch(
      `/api/nifti/${radarUploadId}/slice?idx=${idx}&wc=${WIN.wc}&ww=${WIN.ww}`);
    if (!r.ok) throw new Error(String(r.status));
    const blob = await r.blob();
    if (sliceUrl) URL.revokeObjectURL(sliceUrl);
    sliceUrl = URL.createObjectURL(blob);
    $("slice").src = sliceUrl;
    $("slice-info").textContent = `${idx} / ${r.headers.get("X-Total-Slices")}`;
  } catch (e) {
    $("slice-info").textContent = "slice unavailable";
  }
}

let sliceDebounce = null;
$("slice-range").oninput = () => {
  syncSliceNum();
  clearTimeout(sliceDebounce);
  sliceDebounce = setTimeout(loadSlice, 150);
};
$("slice-num").onchange = () => setSlice(Number($("slice-num").value) || 0);
$("jump-first").onclick = () => setSlice(0);
$("jump-last").onclick = () => setSlice(Number($("slice-range").max) || 0);

document.querySelectorAll(".win-preset").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".win-preset").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    WIN.wc = Number(b.dataset.wc);
    WIN.ww = Number(b.dataset.ww);
    resetZoom();
    loadSlice();
  };
});

document.addEventListener("keydown", (e) => {
  if ($("view-radar").classList.contains("hidden")) return;
  const ae = document.activeElement;
  if (ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA")) return;
  if (!radarUploadId) return;
  const cur = Number($("slice-range").value) || 0;
  switch (e.key) {
    case "ArrowLeft":  e.preventDefault(); setSlice(cur - 1); break;
    case "ArrowRight": e.preventDefault(); setSlice(cur + 1); break;
    case "PageUp":     e.preventDefault(); setSlice(cur - 10); break;
    case "PageDown":   e.preventDefault(); setSlice(cur + 10); break;
    case "Home":       e.preventDefault(); setSlice(0); break;
    case "End":        e.preventDefault(); setSlice(Number($("slice-range").max) || 0); break;
  }
});

/* ---------------- RADAR: job polling ---------------- */
async function pollRadar(jobId) {
  const token = ++radarPollToken;
  radarAbort = new AbortController();
  const signal = radarAbort.signal;
  for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
    if (token !== radarPollToken) return;
    let j;
    try {
      j = await fetchJSON(`/api/radar/${jobId}`, { signal });
    } catch (e) {
      if (e.name === "AbortError") return;
      setFindingState("empty");
      toast("err", "Could not fetch RADAR status", e.message);
      return;
    }
    if (j.status === "error") {
      setFindingState("empty");
      toast("err", "RADAR error", j.error || "unknown error");
      return;
    }
    if (j.status === "done") {
      radarFindings = Array.isArray(j.findings) ? j.findings : [];
      setFindingState("table");
      renderFindings();
      $("csv").disabled = false;
      $("csv").onclick = () => { if (j.csv_url) window.open(j.csv_url); };
      toast("ok", "Analysis complete", `${radarFileName} · ${radarFindings.length} findings`);
      return;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL));
  }
  setFindingState("empty");
  toast("err", "RADAR timed out", "The job did not finish in time; watch the queue in the status bar.");
}

/* ---------------- Findings table ---------------- */
function tierOf(score) {
  return score >= 0.7 ? "high" : score >= 0.3 ? "mid" : "low";
}

function organOf(f) {
  const name = f.name || "";
  const i = name.indexOf("_");
  return i > 0 ? name.slice(0, i) : "Other";
}
function shortOf(f) {
  const name = f.name || "";
  const i = name.indexOf("_");
  return i > 0 ? name.slice(i + 1) : name;
}

function filteredFindings() {
  const thr = parseFloat($("threshold").value);
  const q = $("search").value.toLowerCase().trim();
  return radarFindings
    .filter((f) => typeof f.score === "number" && f.score >= thr
      && (f.name || "").toLowerCase().includes(q))
    .sort((a, b) => sortDesc ? b.score - a.score : a.score - b.score);
}

function scoreCell(score) {
  const td = document.createElement("td");
  td.className = "f-score";
  const tier = tierOf(score);
  const val = document.createElement("div");
  val.className = "score-val score-" + (tier === "high" ? "high" : tier === "mid" ? "mid" : "low");
  val.textContent = score.toFixed(3);
  const bar = document.createElement("div");
  bar.className = "bar bar-" + tier;
  const fill = document.createElement("i");
  fill.style.width = Math.round(Math.max(2, score * 100)) + "%";
  bar.appendChild(fill);
  td.append(val, bar);
  return td;
}

function findingRow(f) {
  const tr = document.createElement("tr");
  tr.className = "finding-row";
  const nameTd = document.createElement("td");
  nameTd.textContent = shortOf(f);
  nameTd.title = f.label || f.name;
  const organTd = document.createElement("td");
  organTd.className = "f-organ";
  organTd.textContent = organOf(f);
  tr.append(nameTd, organTd, scoreCell(f.score));
  return tr;
}

function groupRow(name, count) {
  const tr = document.createElement("tr");
  tr.className = "group-row";
  const td = document.createElement("td");
  td.colSpan = 3;
  td.appendChild(document.createTextNode(name + " "));
  const span = document.createElement("span");
  span.className = "g-count";
  span.textContent = count + (count === 1 ? " finding" : " findings");
  td.appendChild(span);
  tr.appendChild(td);
  return tr;
}

function renderTopChips(rows) {
  const box = $("top-findings");
  box.replaceChildren();
  rows.slice(0, 6).forEach((f) => {
    const tier = tierOf(f.score);
    const chip = document.createElement("span");
    chip.className = "chip " + tier;
    chip.title = f.label || f.name;
    const nm = document.createElement("span");
    nm.textContent = shortOf(f);
    const sc = document.createElement("span");
    sc.className = "mono";
    sc.textContent = f.score.toFixed(2);
    chip.append(nm, sc);
    box.appendChild(chip);
  });
}

function renderFindings() {
  const rows = filteredFindings();
  const thr = parseFloat($("threshold").value);
  $("findings-count").textContent = `${rows.length} / ${radarFindings.length}`;
  $("findings-count").title = `${rows.length} findings ≥ ${thr.toFixed(2)}`;
  renderTopChips(rows);
  setFindingState("table");

  const tbody = $("findings");
  tbody.replaceChildren();

  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.className = "empty";
    td.colSpan = 3;
    td.textContent = `No findings above the threshold (${thr.toFixed(2)})`;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  if ($("group-toggle").checked) {
    const groups = new Map();
    rows.forEach((f) => {
      const g = organOf(f);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(f);
    });
    [...groups.entries()]
      .sort((a, b) => Math.max(...b[1].map((x) => x.score)) - Math.max(...a[1].map((x) => x.score)))
      .forEach(([g, list]) => {
        tbody.appendChild(groupRow(g, list.length));
        list.forEach((f) => tbody.appendChild(findingRow(f)));
      });
  } else {
    rows.forEach((f) => tbody.appendChild(findingRow(f)));
  }
}

$("threshold").oninput = () => {
  $("thr-val").textContent = parseFloat($("threshold").value).toFixed(2);
  document.querySelectorAll(".thr-preset").forEach((b) =>
    b.classList.toggle("on", Math.abs(Number(b.dataset.thr) - Number($("threshold").value)) < 0.005));
  renderFindings();
};
document.querySelectorAll(".thr-preset").forEach((b) => {
  b.onclick = () => {
    $("threshold").value = b.dataset.thr;
    $("threshold").oninput();
  };
});
$("search").oninput = renderFindings;
$("group-toggle").onchange = renderFindings;
$("sort-score").onclick = () => {
  sortDesc = !sortDesc;
  $("sort-arrow").textContent = sortDesc ? "▾" : "▴";
  renderFindings();
};

/* ---------------- Chat (ClinFusion) ---------------- */
let chatHistory = [];
let chatAttachments = [];
let chatUploading = false;
let sending = false;
let chatPollToken = 0;
let chatAbort = null;
let chatUploadToken = 0;

function niftiThumb(up) {
  if (!up.upload_id || !up.n_slices) return null;
  const mid = Math.floor(up.n_slices / 2);
  return `/api/nifti/${up.upload_id}/slice?idx=${mid}&wc=0&ww=2000`;
}

function hideChatEmpty() {
  const el = $("chat-empty");
  if (el) el.classList.add("hidden");
}

$("chat-file").onchange = async () => {
  if (chatUploading) return;
  const files = Array.from($("chat-file").files);
  $("chat-file").value = "";
  if (!files.length) return;
  const token = chatUploadToken;
  chatUploading = true;
  $("send").disabled = true;
  try {
    const uploaded = [];
    for (const f of files) {
      const fd = new FormData();
      fd.append("file", f);
      const up = await fetchJSON("/api/upload", { method: "POST", body: fd });
      if (up.upload_id) {
        const isImage = (up.kind === "image") || /\.(png|jpe?g)$/i.test(f.name);
        const url = isImage ? URL.createObjectURL(f) : null;
        uploaded.push({
          id: up.upload_id,
          name: up.filename || f.name,
          kind: up.kind || "nifti",
          n_slices: up.n_slices || null,
          url,
          thumb: url || niftiThumb(up),
          sent: false,
        });
      }
    }
    if (token === chatUploadToken) {
      chatAttachments = chatAttachments.concat(uploaded);
      renderChatAttachments();
    }
  } catch (e) {
    toast("err", "Attachment upload failed", e.message);
  } finally {
    chatUploading = false;
    $("send").disabled = false;
  }
};

function renderChatAttachments() {
  const box = $("att-list");
  if (!box) return;
  box.replaceChildren();
  $("att-count").textContent = String(chatAttachments.length);

  if (!chatAttachments.length) {
    const e = document.createElement("div");
    e.className = "empty att-empty";
    const t = document.createElement("span");
    t.className = "empty-title";
    t.textContent = "No attachments yet";
    const s = document.createElement("span");
    s.className = "empty-sub";
    s.textContent = "Use 📎 to add 2D images (.jpg/.png) or a 3D CT (.nii.gz); previews appear here.";
    e.append(t, s);
    box.appendChild(e);
    return;
  }

  chatAttachments.forEach((a, idx) => {
    const card = document.createElement("div");
    card.className = "att-card";
    card.title = "Click for full size";

    if (a.thumb) {
      const thumb = document.createElement("img");
      thumb.className = "att-thumb";
      thumb.alt = a.name;
      thumb.src = a.thumb;
      thumb.onclick = () => openImage(a.thumb);
      card.appendChild(thumb);
    } else {
      const ph = document.createElement("div");
      ph.className = "att-thumb att-thumb-ph";
      ph.textContent = "no preview";
      card.appendChild(ph);
    }

    const meta = document.createElement("div");
    meta.className = "att-meta";
    const name = document.createElement("div");
    name.className = "att-name";
    name.textContent = a.name;
    name.title = a.name;
    const sub = document.createElement("div");
    sub.className = "att-sub";
    const badge = document.createElement("span");
    badge.className = "att-badge" + (a.kind === "nifti" ? " v3" : "");
    badge.textContent = a.kind === "nifti" ? "3D CT" : "2D";
    sub.appendChild(badge);
    if (a.kind === "nifti" && a.n_slices) {
      const n = document.createElement("span");
      n.className = "mono";
      n.textContent = a.n_slices + (a.n_slices === 1 ? " slice" : " slices");
      sub.appendChild(n);
    }
    if (a.sent) {
      const ok = document.createElement("span");
      ok.className = "mono";
      ok.textContent = "✓ sent";
      sub.appendChild(ok);
    }
    meta.append(name, sub);
    card.appendChild(meta);

    if (!a.sent) {
      const rm = document.createElement("button");
      rm.className = "att-rm";
      rm.textContent = "×";
      rm.title = "Remove";
      rm.onclick = (e) => {
        e.stopPropagation();
        if (a.url) URL.revokeObjectURL(a.url);
        chatAttachments.splice(idx, 1);
        renderChatAttachments();
      };
      card.appendChild(rm);
    }
    box.appendChild(card);
  });
}

/* Markdown-lite: builds DOM safely (no innerHTML) */
function mdInline(text) {
  const frag = document.createDocumentFragment();
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tok = m[0];
    if (tok.startsWith("**")) {
      const b = document.createElement("strong");
      b.textContent = tok.slice(2, -2);
      frag.appendChild(b);
    } else {
      const c = document.createElement("code");
      c.textContent = tok.slice(1, -1);
      frag.appendChild(c);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}

function mdRender(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg-text";
  const blocks = String(text || "").split(/```/);
  blocks.forEach((blk, bi) => {
    if (bi % 2 === 1) {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = blk.replace(/^[a-zA-Z0-9]*\n/, "");
      pre.appendChild(code);
      wrap.appendChild(pre);
      return;
    }
    blk.split(/\n{2,}/).forEach((para) => {
      const lines = para.split("\n").filter((l) => l.trim() !== "");
      if (!lines.length) return;
      if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
        const ul = document.createElement("ul");
        lines.forEach((l) => {
          const li = document.createElement("li");
          li.appendChild(mdInline(l.replace(/^\s*[-*]\s+/, "")));
          ul.appendChild(li);
        });
        wrap.appendChild(ul);
      } else if (lines.length > 1 && lines.every((l) => l.includes("|"))) {
        const table = document.createElement("table");
        lines.forEach((l, ri) => {
          const tr = document.createElement("tr");
          l.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").forEach((cellRaw) => {
            const cell = document.createElement(ri === 0 ? "th" : "td");
            cell.appendChild(mdInline(cellRaw.trim()));
            tr.appendChild(cell);
          });
          table.appendChild(tr);
        });
        wrap.appendChild(table);
      } else {
        const p = document.createElement("p");
        p.appendChild(mdInline(lines.join(" ")));
        wrap.appendChild(p);
      }
    });
  });
  return wrap;
}

function addMsg(role, text, attachments = []) {
  hideChatEmpty();
  const box = document.createElement("div");
  box.className = "msg " + (role === "user" ? "msg-user" : "msg-assistant");

  const head = document.createElement("div");
  head.className = "msg-head";
  const who = document.createElement("span");
  who.textContent = role === "user" ? "You" : "ClinFusion-32B";
  head.appendChild(who);
  if (role !== "user") {
    const cp = document.createElement("button");
    cp.className = "btn btn-icon";
    cp.textContent = "Copy";
    cp.onclick = async () => {
      try {
        await navigator.clipboard.writeText(text);
        toast("ok", "Copied", "");
      } catch (e) {
        toast("err", "Copy failed", e.message);
      }
    };
    head.appendChild(cp);
  }
  box.appendChild(head);

  box.appendChild(mdRender(text));

  const refs = attachments || [];
  if (refs.length) {
    const wrap = document.createElement("div");
    wrap.className = "msg-refs";
    refs.forEach((a) => {
      const chip = document.createElement("span");
      chip.className = "att-ref" + (a.thumb ? " clickable" : "");
      chip.textContent = "📎 " + a.name;
      chip.title = a.name;
      if (a.thumb) chip.onclick = () => openImage(a.thumb);
      wrap.appendChild(chip);
    });
    box.appendChild(wrap);
  }

  $("messages").appendChild(box);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function makeThinking() {
  const el = document.createElement("div");
  el.className = "thinking";
  el.textContent = "thinking ";
  ["", "", ""].forEach(() => el.appendChild(document.createElement("i")));
  $("messages").appendChild(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function cancelChatPolling() {
  chatPollToken++;
  if (chatAbort) {
    chatAbort.abort();
    chatAbort = null;
  }
}

$("new-chat").onclick = async () => {
  cancelChatPolling();
  chatUploadToken++;
  try {
    await fetchJSON("/api/chat/reset", { method: "POST" });
  } catch (e) {
    toast("err", "Could not reset chat", e.message);
  }
  chatHistory = [];
  chatAttachments.forEach((a) => { if (a.url) URL.revokeObjectURL(a.url); });
  chatAttachments = [];
  sending = false;
  $("messages").replaceChildren();
  if ($("chat-empty")) $("messages").appendChild($("chat-empty")), $("chat-empty").classList.remove("hidden");
  renderChatAttachments();
  $("chat-input").value = "";
  $("send").disabled = false;
};

async function pollChat(jobId, prompt, thinking) {
  const token = ++chatPollToken;
  chatAbort = new AbortController();
  const signal = chatAbort.signal;
  for (let i = 0; i < CHAT_POLL_MAX_ATTEMPTS; i++) {
    if (token !== chatPollToken) return;
    let r;
    try {
      r = await fetchJSON(`/api/chat/${jobId}`, { signal });
    } catch (e) {
      if (e.name === "AbortError") return;
      if (thinking.isConnected) thinking.remove();
      toast("err", "Chat error", e.message);
      return;
    }
    if (r.status === "error") {
      if (thinking.isConnected) thinking.remove();
      toast("err", "ClinFusion error", r.error || "unknown error");
      return;
    }
    if (r.status === "done") {
      if (thinking.isConnected) thinking.remove();
      addMsg("assistant", r.reply);
      chatHistory.push({ role: "user", text: prompt });
      chatHistory.push({ role: "assistant", text: r.reply });
      return;
    }
    await new Promise((res) => setTimeout(res, POLL_INTERVAL));
  }
  if (thinking.isConnected) thinking.remove();
  toast("err", "Chat timed out", "Generation did not complete in time.");
}

async function send() {
  if (sending || chatUploading) return;
  const prompt = $("chat-input").value.trim();
  if (!prompt) return;
  sending = true;
  $("send").disabled = true;
  addMsg("user", prompt, chatAttachments);
  $("chat-input").value = "";
  $("chat-input").style.height = "";
  const body = {
    prompt,
    history: chatHistory,
    attachment_ids: chatAttachments.map((a) => a.id),
  };
  chatAttachments.forEach((a) => { a.sent = true; });
  renderChatAttachments();
  const thinking = makeThinking();
  try {
    const j = await fetchJSON("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await pollChat(j.job_id, prompt, thinking);
  } catch (e) {
    if (thinking.isConnected) thinking.remove();
    toast("err", "Could not send", e.message);
  } finally {
    sending = false;
    $("send").disabled = false;
  }
}

$("send").onclick = send;
$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
$("chat-input").addEventListener("input", () => {
  const el = $("chat-input");
  el.style.height = "auto";
  el.style.height = Math.min(130, el.scrollHeight) + "px";
});

/* Initial render */
setFindingState("empty");
applyTransform();