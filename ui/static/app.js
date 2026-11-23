"use strict";

const NAMES = {
  lips: "Lips", blush: "Blush", shadow: "Eyeshadow", liner: "Eyeliner", lashes: "Lashes", skin: "Foundation",
  lens: "Lenses", brows: "Brows", teeth: "Teeth", hair: "Hair colour", mustache: "Mustache", beard: "Beard",
};
const TABS = [
  ["Face", ["skin", "blush", "brows"]],
  ["Eyes", ["shadow", "liner", "lashes", "lens"]],
  ["Lips", ["lips", "teeth"]],
  ["Hair", ["hair", "mustache", "beard"]],
];
const AMOUNT_LABEL = { skin: "Cover" };

const $ = (id) => document.getElementById(id);
const els = {
  frame: $("frame"), stage: $("stage"), before: $("before"), after: $("after"), divider: $("divider"), empty: $("empty"),
  noFace: $("noFace"), flash: $("flash"), fps: $("fps"), compareBtn: $("compareBtn"), snapBtn: $("snapBtn"),
  modeCam: $("modeCam"), modePhoto: $("modePhoto"), video: $("video"), tagBefore: $("tagBefore"), tagAfter: $("tagAfter"),
  lookSelect: $("lookSelect"), deleteLookBtn: $("deleteLookBtn"), saveForm: $("saveForm"), lookName: $("lookName"),
  tabs: $("tabs"), sections: $("sections"), unet: $("unetToggle"),
};
const grab = document.createElement("canvas");

let S = null;
let tab = 2;
let mode = null;  // null | "camera" | "photo"
let loopId = 0, stream = null;
let photoBlob = null, photoBusy = false, photoDirty = false;
let compare = false, holding = false, hasPicture = false;
let deleteArmed = null;

async function api(action, extra = {}, { quiet = true } = {}) {
  try {
    const r = await fetch("/api/do", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }) });
    const d = await r.json();
    if (d.state) { S = d.state; render(); }
    if (d.msg && (!quiet || !d.ok) && !d.exists) toast(d.msg, !d.ok);
    if (mode === "photo") reprocessPhoto();
    return d;
  } catch {
    toast("Lost the server. Is ui/server.py still running?", true);
    return { ok: false };
  }
}

// sliders fire many values: send only the newest, one request at a time
const pending = new Map();
function sendLatest(key, action, extra) {
  const first = !pending.has(key);
  pending.set(key, { action, extra });
  if (!first) return;
  (async () => {
    while (pending.get(key)) {
      const job = pending.get(key);
      pending.set(key, null);
      await api(job.action, job.extra);
    }
    pending.delete(key);
  })();
}

async function sendFrame(blob, track) {
  const r = await fetch(`/api/frame?track=${track ? 1 : 0}`, { method: "POST", body: blob });
  if (!r.ok) throw new Error(await r.text());
  return { face: r.headers.get("X-Face") === "1", bmp: await createImageBitmap(await r.blob()) };
}

function showResult({ face, bmp }) {
  for (const c of [els.before, els.after]) {
    if (c.width !== grab.width || c.height !== grab.height) { c.width = grab.width; c.height = grab.height; }
  }
  els.before.getContext("2d").drawImage(grab, 0, 0);
  els.after.getContext("2d").drawImage(bmp, 0, 0, grab.width, grab.height);
  els.noFace.hidden = face;
  if (!hasPicture) {
    hasPicture = true;
    els.empty.hidden = true;
    els.compareBtn.disabled = els.snapBtn.disabled = false;
  }
}

let serverCamera = null, serverFrame = null;

async function startCamera() {
  if (serverCamera === null) {
    try { serverCamera = (await (await fetch("/api/config")).json()).server_camera; } catch { serverCamera = false; }
  }
  if (serverCamera) {
    try { serverFrame = await fetchServerFrame(); } catch (e) { toast(`Couldn't open the server camera: ${e.message}`, true); return; }
    mode = "camera";
    photoBlob = null;
    els.stage.style.aspectRatio = `${serverFrame.width} / ${serverFrame.height}`;
    setModeButtons();
    cameraLoop(++loopId);
    return;
  }
  try {
    stream = stream || await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
  } catch {
    toast("Couldn't open the camera. Allow camera access in the browser, or choose a photo.", true);
    return;
  }
  els.video.srcObject = stream;
  await els.video.play();
  mode = "camera";
  photoBlob = null;
  els.stage.style.aspectRatio = `${els.video.videoWidth || 4} / ${els.video.videoHeight || 3}`;
  setModeButtons();
  cameraLoop(++loopId);
}

async function fetchServerFrame() {
  const r = await fetch("/api/camera/raw", { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return createImageBitmap(await r.blob());
}

function stopCamera() {
  loopId++;
  if (serverCamera) fetch("/api/camera/release", { method: "POST" }).catch(() => {});
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  els.fps.textContent = "";
}

async function cameraLoop(id) {
  const ctx = grab.getContext("2d");
  let fps = null, last = performance.now(), errors = 0;
  while (id === loopId) {
    let v = els.video;
    try {
      if (serverCamera) v = serverFrame = await fetchServerFrame();
    } catch {
      if (++errors === 3) toast("Lost the server camera. Is ui/server.py still running?", true);
      await sleep(800);
      continue;
    }
    const vw = v.videoWidth || v.width, vh = v.videoHeight || v.height;
    if (!serverCamera && (v.readyState < 2 || !vw)) { await sleep(30); continue; }
    const scale = Math.min(1, 640 / vw);
    const w = Math.round(vw * scale), h = Math.round(vh * scale);
    if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
    ctx.save(); ctx.translate(w, 0); ctx.scale(-1, 1); ctx.drawImage(v, 0, 0, w, h); ctx.restore(); // mirror
    try {
      const blob = await new Promise((res) => grab.toBlob(res, "image/jpeg", 0.85));
      const out = await sendFrame(blob, true);
      if (id !== loopId) break;
      showResult(out);
      errors = 0;
    } catch {
      if (++errors === 3) toast("Lost the server. Is ui/server.py still running?", true);
      await sleep(800);
      continue;
    }
    const now = performance.now(), f = 1000 / Math.max(now - last, 1);
    fps = fps === null ? f : 0.9 * fps + 0.1 * f;
    last = now;
    els.fps.textContent = `${fps.toFixed(0)} fps`;
  }
}

async function loadPhoto(file) {
  if (!file || !file.type.startsWith("image/")) { toast("That file isn't an image.", true); return; }
  let bmp;
  try { bmp = await createImageBitmap(file); } catch { toast("Couldn't open that image.", true); return; }
  stopCamera();
  mode = "photo";
  setModeButtons();
  const scale = Math.min(1, 1280 / Math.max(bmp.width, bmp.height));
  grab.width = Math.round(bmp.width * scale); grab.height = Math.round(bmp.height * scale);
  grab.getContext("2d").drawImage(bmp, 0, 0, grab.width, grab.height);
  els.stage.style.aspectRatio = `${grab.width} / ${grab.height}`;
  photoBlob = await new Promise((res) => grab.toBlob(res, "image/jpeg", 0.92));
  reprocessPhoto(true);
}

async function reprocessPhoto(fresh = false) {
  if (!photoBlob) return;
  if (photoBusy) { photoDirty = true; return; }
  photoBusy = true;
  try {
    do {
      photoDirty = false;
      const out = await sendFrame(photoBlob, false);
      if (mode !== "photo") break;
      showResult(out);
      if (fresh && !out.face) toast("No face found in this photo. Try one where the face is larger and facing the camera.", true);
      fresh = false;
    } while (photoDirty);
  } catch {
    toast("Lost the server. Is ui/server.py still running?", true);
  } finally {
    photoBusy = false;
  }
}

function setModeButtons() {
  els.modeCam.classList.toggle("on", mode === "camera");
  els.modePhoto.classList.toggle("on", mode === "photo");
}

function render() {
  if (!S) return;
  renderLooks();
  renderTabs();
  if (!holding) renderSections();
  els.unet.checked = S.unet;
  els.unet.disabled = !S.has_unet;
  for (const part of ["box", "dots", "label"]) detectEls[part].checked = !!(S.show && S.show[part]);
}
const detectEls = { box: $("showBox"), dots: $("showDots"), label: $("showLabel") };
for (const part of Object.keys(detectEls)) {
  detectEls[part].onchange = () => api("detect", { part, on: detectEls[part].checked });
}

function renderLooks() {
  const sel = els.lookSelect;
  sel.innerHTML = "";
  const groups = [["Built in", S.looks.filter((n) => S.builtin.includes(n))], ["Mine", S.looks.filter((n) => !S.builtin.includes(n))]];
  for (const [label, names] of groups) {
    if (!names.length) continue;
    const g = document.createElement("optgroup");
    g.label = label;
    for (const n of names) g.appendChild(new Option(cap(n), n, false, n === S.look));
    sel.appendChild(g);
  }
  sel.value = S.look;
  els.deleteLookBtn.disabled = S.builtin.includes(S.look);
}

function renderTabs() {
  els.tabs.innerHTML = "";
  TABS.forEach(([label, names], i) => {
    const on = names.filter((n) => layer(n).on).length;
    const b = document.createElement("button");
    b.className = "tab" + (i === tab ? " on" : "");
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === tab);
    b.innerHTML = `${label}<span class="count">${on}/${names.length}</span>`;
    b.title = `${on} of ${names.length} on`;
    b.onclick = () => { tab = i; render(); };
    els.tabs.appendChild(b);
  });
}

function renderSections() {
  els.sections.innerHTML = "";
  for (const name of TABS[tab][1]) els.sections.appendChild(section(layer(name)));
}

function section(L) {
  const sec = el("div", "section" + (L.on ? "" : " off"));
  const head = el("div", "sec-head");
  head.appendChild(el("span", "sec-name", NAMES[L.name] || cap(L.name)));
  const sw = el("label", "toggle-row");
  sw.title = L.on ? "Turn off" : "Turn on";
  sw.innerHTML = `<input type="checkbox" ${L.on ? "checked" : ""} aria-label="${NAMES[L.name]}"><span class="toggle"></span>`;
  sw.querySelector("input").onchange = () => api("toggle", { layer: L.name });
  head.appendChild(sw);
  sec.appendChild(head);
  if (!L.on) return sec;

  const body = el("div", "sec-body");
  const pans = el("div", "pans");
  const preset = L.presets.find((p) => p.hex.toUpperCase() === L.color.toUpperCase());
  for (const p of L.presets) {
    const b = el("button", "pan" + (p === preset ? " sel" : ""));
    b.title = p.name;
    b.innerHTML = `<span class="chip" style="background:${p.hex}"></span><span class="name">${p.name}</span>`;
    b.onclick = () => api("color", { layer: L.name, hex: p.hex });
    pans.appendChild(b);
  }
  if (!preset) {
    const b = el("div", "pan sel");
    b.innerHTML = `<span class="chip" style="background:${L.color}"></span><span class="name">custom</span>`;
    pans.appendChild(b);
  }
  const add = el("label", "pan add");
  add.title = "Pick any colour";
  add.innerHTML = `<span class="chip"><svg><use href="#i-plus"/></svg></span><span class="name">other</span><input type="color" value="${L.color.toLowerCase()}">`;
  const ci = add.querySelector("input");
  ci.addEventListener("click", () => { holding = true; });   // don't rebuild the picker while it's open
  ci.addEventListener("input", () => sendLatest("color", "color", { layer: L.name, hex: ci.value.toUpperCase() }));
  ci.addEventListener("change", release);
  ci.addEventListener("blur", release);
  pans.appendChild(add);
  body.appendChild(pans);

  if (L.styles.length) {
    const f = el("div", "field");
    f.appendChild(el("span", "label", "Style"));
    const row = el("div", "styles");
    for (const st of L.styles) {
      const b = el("button", "style" + (st === L.style ? " on" : ""), st);
      b.onclick = () => api("style", { layer: L.name, style: st });
      row.appendChild(b);
    }
    f.appendChild(row);
    body.appendChild(f);
  }

  body.appendChild(slider(AMOUNT_LABEL[L.name] || "Amount", L.k, (v) => sendLatest(`k-${L.name}`, "strength", { layer: L.name, value: v })));
  if (L.extra) {
    body.appendChild(slider(L.extra.label, L.extra.value,
      (v) => sendLatest(`x-${L.name}`, "extra", { layer: L.name, key: L.extra.key, value: v })));
  }
  sec.appendChild(body);
  return sec;
}

function slider(label, value, onValue) {
  const f = el("label", "field");
  f.innerHTML = `<span class="label">${label}</span><input type="range" min="0" max="100" step="1" value="${Math.round(value * 100)}"><span class="val"></span>`;
  const r = f.querySelector("input"), val = f.querySelector(".val");
  const paint = () => { r.style.setProperty("--p", `${r.value}%`); val.textContent = r.value; };
  paint();
  r.addEventListener("pointerdown", () => { holding = true; });
  r.addEventListener("input", () => { paint(); onValue(r.value / 100); });
  r.addEventListener("change", release);
  r.addEventListener("pointerup", () => { holding = false; });
  return f;
}

function release() { if (holding) { holding = false; render(); } }

els.lookSelect.onchange = () => api("look", { name: els.lookSelect.value });
$("resetBtn").onclick = () => api("reset", {}, { quiet: false });
$("bareBtn").onclick = () => api("bare", {}, { quiet: false });
$("saveLookBtn").onclick = () => {
  els.saveForm.hidden = false;
  els.lookName.value = S && !S.builtin.includes(S.look) ? S.look : "";
  els.lookName.focus(); els.lookName.select();
};
$("saveCancel").onclick = () => { els.saveForm.hidden = true; };
els.lookName.addEventListener("keydown", (e) => { if (e.key === "Escape") els.saveForm.hidden = true; });
els.saveForm.onsubmit = async (e) => {
  e.preventDefault();
  const name = els.lookName.value;
  let d = await api("save_look", { name }, { quiet: false });
  if (d.exists) {
    if (!confirm(`Replace your look "${name.trim()}"?`)) return;
    d = await api("save_look", { name, overwrite: true }, { quiet: false });
  }
  if (d.ok) els.saveForm.hidden = true;
};
els.deleteLookBtn.onclick = () => {
  const b = els.deleteLookBtn, label = b.querySelector("span");
  if (deleteArmed) {
    clearTimeout(deleteArmed); deleteArmed = null;
    b.classList.remove("warn"); label.textContent = "Delete";
    api("delete_look", { name: S.look }, { quiet: false });
    return;
  }
  b.classList.add("warn"); label.textContent = "Click again to delete";
  deleteArmed = setTimeout(() => { deleteArmed = null; b.classList.remove("warn"); label.textContent = "Delete"; }, 3000);
};
els.unet.onchange = () => api("unet", {}, { quiet: false });

function setCompare(on) {
  compare = on;
  els.compareBtn.classList.toggle("on", on);
  els.divider.hidden = els.tagBefore.hidden = els.tagAfter.hidden = !on;
  els.stage.style.setProperty("--split", on ? "50%" : "0%");
}
function moveSplit(x) {
  const r = els.stage.getBoundingClientRect();
  els.stage.style.setProperty("--split", `${Math.min(100, Math.max(0, ((x - r.left) / r.width) * 100))}%`);
}
let splitting = false;
els.stage.addEventListener("pointerdown", (e) => {
  if (!compare) return;
  splitting = true; els.stage.setPointerCapture(e.pointerId); moveSplit(e.clientX);
});
els.stage.addEventListener("pointermove", (e) => { if (splitting) moveSplit(e.clientX); });
els.stage.addEventListener("pointerup", () => { splitting = false; });

async function snap() {
  if (!hasPicture) return;
  els.flash.classList.remove("go"); void els.flash.offsetWidth; els.flash.classList.add("go");
  const d = await api("save_image");
  if (d.ok) toast(d.msg, false, { href: els.after.toDataURL("image/png"), text: "Download", name: "blush-booth.png" });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const layer = (name) => S.layers.find((l) => l.name === name);
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function toast(text, bad = false, link = null) {
  const t = el("div", "toast" + (bad ? " bad" : ""), text);
  if (link) {
    const a = el("a", "", link.text);
    a.href = link.href; a.download = link.name;
    t.appendChild(a);
  }
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), link ? 6000 : 3000);
}

$("startCam").onclick = startCamera;
els.modeCam.onclick = startCamera;
for (const id of ["filePick", "filePick2"]) $(id).onchange = (e) => { loadPhoto(e.target.files[0]); e.target.value = ""; };
els.compareBtn.onclick = () => setCompare(!compare);
els.snapBtn.onclick = snap;

["dragenter", "dragover"].forEach((t) => els.frame.addEventListener(t, (e) => { e.preventDefault(); els.frame.classList.add("dragging"); }));
["dragleave", "drop"].forEach((t) => els.frame.addEventListener(t, (e) => { e.preventDefault(); els.frame.classList.remove("dragging"); }));
els.frame.addEventListener("drop", (e) => loadPhoto(e.dataTransfer.files[0]));

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input[type=text], select") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === " ") { e.preventDefault(); snap(); }
  else if (e.key.toLowerCase() === "c" && hasPicture) setCompare(!compare);
  else if (/^[1-4]$/.test(e.key) && S) { tab = Number(e.key) - 1; render(); }
});

(async () => {
  try {
    const d = await (await fetch("/api/state")).json();
    S = d.state;
    render();
    if (d.msg) toast(d.msg, true);
  } catch { toast("Can't reach the server. Start it with: python ui/server.py", true); }
})();
