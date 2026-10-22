// 3D incident view: Copernicus DEM relief of Bajo Piura / Sechura, a Sentinel-2 photo per
// acquisition (raw or cloud-free composite) and the AI water per analysed date, driven by a
// timeline player. Data from scripts/system/terrain_export.py and terrain_frames.py
// (app/public/terrain/<zone>/); key events and facts from ../config.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EVENTS, type FloodEvent, type KeyEvent } from "../config";
import "./terrain.css";

interface Spread { from: string; days: number; new_km2: number; receded_km2: number; net_km2: number; net_km2_per_day: number; observed_share: number }
interface Analysis {
  date: string; event: string; label: string; is_pre: boolean; water_km2: number; nodata_share: number; comparable: boolean;
  flooded_km2?: number; flood_observed_share?: number; spread?: Spread;
  reference?: { source: string; pixels: number; iou: number; precision: number; recall: number };
}
interface Meta {
  title: string; size_m: [number, number]; grid: number; bounds_wgs84: [number, number, number, number];
  height_min: number; height_max: number; dates: Analysis[]; pre_event: Record<string, string | null>;
}
interface Frame { date: string; year: string; cloud: number; valid: number; clear: boolean; fresh_share: number; composite_clear_share: number; composite_median_age_days: number | null; composite_p90_age_days: number | null }
interface Weather {
  precip: { source: string; note: string }; discharge: { source: string; note: string };
  years: Record<string, { days: string[]; precip_mm: (number | null)[]; discharge_m3s: (number | null)[] }>;
}

const BASE = `${import.meta.env.BASE_URL}terrain/piura/`;
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const DAY = 86_400_000;
const toDay = (iso: string) => Math.round(Date.parse(`${iso}T00:00:00Z`) / DAY);
const toIso = (d: number) => new Date(d * DAY).toISOString().slice(0, 10);
const MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct", "nov", "dic"];
const MONTHS_LONG = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "setiembre", "octubre", "noviembre", "diciembre"];
const DOW = ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"];
const fmtDay = (d: number, year = true) => { const t = new Date(d * DAY); return `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]}${year ? ` ${t.getUTCFullYear()}` : ""}`; };
const km2 = (v: number) => `${v.toLocaleString("es-PE", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} km²`;
const pct = (v: number) => `${Math.round(v * 100)} %`;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

const YEARS: Record<string, { event: FloodEvent; kicker: string; start: string; end: string; initial: string; test: { chips: number; iou: number; precision: number; recall: number; set: string } }> = {
  "2023": { event: EVENTS.yaku2023, kicker: "Ciclón Yaku · 2023", start: "2023-01-01", end: "2023-05-31", initial: "2023-03-22",
    // outputs/system_eval/rf/rf5_chips.json (= outputs/phaseD/test/peru-test_groups.csv, peru/unet, full)
    test: { chips: 172, iou: 0.828, precision: 0.966, recall: 0.853, set: "peru-test" } },
  "2017": { event: EVENTS.nino2017, kicker: "Niño Costero · 2017", start: "2017-01-01", end: "2017-05-31", initial: "2017-04-07",
    // outputs/system_eval/rf/rf5_chips.json (= peru-2017-test_groups.csv, no_2023_train_overlap)
    test: { chips: 101, iou: 0.878, precision: 1.0, recall: 0.878, set: "peru-2017-test, lejos de 2023" } },
};

async function main() {
  const [meta, frameDoc, weather] = await Promise.all([
    fetch(`${BASE}meta.json`).then((r) => r.json() as Promise<Meta>),
    fetch(`${BASE}frames.json`).then((r) => r.json() as Promise<{ frames: Frame[] }>),
    fetch(`${BASE}weather.json`).then((r) => r.json() as Promise<Weather>),
  ]);
  const heights = new Float32Array(await (await fetch(`${BASE}height.bin`)).arrayBuffer());
  const G = meta.grid, W = meta.size_m[0] / 1000, H = meta.size_m[1] / 1000;

  // ---------------------------------------------------------------- state
  const st = {
    year: "2023", day: toDay(YEARS["2023"].initial), playing: false, speed: 3,
    ground: "composite" as "composite" | "raw" | "relief", rawAll: false, cloudyAI: false,
    showNew: true, showPre: true, showNodata: true,
  };
  const yr = () => YEARS[st.year];
  const range = () => [toDay(yr().start), toDay(yr().end)] as const;
  const framesOf = () => frameDoc.frames.filter((f) => f.year === st.year);
  const usableRaw = (f: Frame) => st.rawAll || (f.valid >= 0.9 && f.cloud <= 0.35);
  const analysesOf = () => meta.dates.filter((a) => a.event === st.year);
  const photoAt = (day: number): Frame | null => {
    const fs = framesOf().filter((f) => st.ground !== "raw" || usableRaw(f));
    let best: Frame | null = null;
    for (const f of fs) if (toDay(f.date) <= day) best = f;
    return best;
  };
  const analysisAt = (day: number): Analysis | null => {
    let best: Analysis | null = null;
    for (const a of analysesOf()) if (toDay(a.date) <= day && (a.comparable || st.cloudyAI)) best = a;
    return best;
  };
  const comparableAt = (day: number): Analysis | null => {
    let best: Analysis | null = null;
    for (const a of analysesOf()) if (toDay(a.date) <= day && a.comparable && !a.is_pre) best = a;
    return best;
  };
  const keyEvents = () => yr().event.events.filter((e) => e.kind !== "image");

  // ---------------------------------------------------------------- renderer, scene, camera
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  let ratio = Math.min(window.devicePixelRatio, 2);
  renderer.setPixelRatio(ratio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  $("scene").appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#e9e2d6");
  const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 500);
  camera.position.set(6, 46, 60);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 2);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.minDistance = 5;
  controls.maxDistance = 120;
  const sunDir = new THREE.Vector3(-40, 26, -25).normalize(); // low sun from the north-west

  // Faint grid on the "table" under the model.
  const grid = new THREE.GridHelper(200, 40, 0xd6cdbf, 0xded6ca);
  grid.position.y = -0.62;
  scene.add(grid);

  // ---------------------------------------------------------------- terrain geometry (x east, z south, row 0 north)
  const geo = new THREE.PlaneGeometry(W, H, G - 1, G - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  let exag = Number(($("exag") as HTMLInputElement).value);
  const applyHeights = () => {
    for (let i = 0; i < G * G; i++) pos.setY(i, (heights[i] / 1000) * exag);
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    $("exag-v").textContent = `×${exag}`;
  };
  applyHeights();
  const colors = new Float32Array(G * G * 3);
  const ramp = [new THREE.Color("#e2d3b3"), new THREE.Color("#d2b88d"), new THREE.Color("#b8966a"), new THREE.Color("#8c6e4f")];
  const span = Math.max(1, meta.height_max - meta.height_min);
  for (let i = 0; i < G * G; i++) {
    const t = Math.min(0.999, Math.max(0, (heights[i] - meta.height_min) / span)) * (ramp.length - 1);
    const c = ramp[Math.floor(t)].clone().lerp(ramp[Math.floor(t) + 1], t % 1);
    colors.set([c.r, c.g, c.b], i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  // ---------------------------------------------------------------- textures (LRU, lazy)
  const loader = new THREE.TextureLoader();
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const texCache = new Map<string, Promise<THREE.Texture>>();
  const inUse = new Set<string>();
  const tex = (url: string, srgb: boolean) => {
    let p = texCache.get(url);
    if (p) { texCache.delete(url); texCache.set(url, p); return p; }
    p = new Promise<THREE.Texture>((res, rej) => loader.load(url, (t) => {
      t.anisotropy = maxAniso;
      if (srgb) t.colorSpace = THREE.SRGBColorSpace;
      res(t);
    }, undefined, rej));
    texCache.set(url, p);
    for (const [k, v] of texCache) {
      if (texCache.size <= 14) break;
      if (inUse.has(k)) continue;
      texCache.delete(k);
      v.then((t) => t.dispose());
    }
    return p;
  };
  const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  blank.needsUpdate = true;
  const photoUrl = (f: Frame) => `${BASE}${st.ground === "raw" ? "frames" : "composite"}/${f.date}.jpg`;
  const waterUrl = (a: Analysis) => `${BASE}water_${a.date}.png`;

  // ---------------------------------------------------------------- materials
  const groundMat = new THREE.ShaderMaterial({
    vertexColors: true,
    uniforms: { uA: { value: blank }, uB: { value: blank }, uT: { value: 1 }, uRelief: { value: 0 }, uSun: { value: sunDir } },
    vertexShader: /* glsl */ `
      varying vec2 vUv; varying vec3 vN; varying vec3 vC;
      void main() { vUv = uv; vN = normal; vC = color; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D uA, uB; uniform float uT, uRelief; uniform vec3 uSun;
      varying vec2 vUv; varying vec3 vN; varying vec3 vC;
      void main() {
        vec3 photo = mix(texture2D(uA, vUv).rgb, texture2D(uB, vUv).rgb, uT);
        vec3 base = mix(photo, vC, uRelief);
        float light = 0.62 + 0.58 * max(dot(normalize(vN), uSun), 0.0);
        gl_FragColor = vec4(base * light, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const ground = new THREE.Mesh(geo, groundMat);
  scene.add(ground);
  const base = new THREE.Mesh(new THREE.BoxGeometry(W, 0.6, H), new THREE.MeshBasicMaterial({ color: "#7a6a57" }));
  base.position.y = -0.33; // top 30 m below sea level: no z-fighting with ground at 0 m
  scene.add(base);

  const waterMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    uniforms: {
      uA: { value: blank }, uB: { value: blank }, uT: { value: 1 }, uPre: { value: blank }, uHasPre: { value: 0 },
      uOn: { value: 0 }, uShowNew: { value: 1 }, uShowPre: { value: 1 }, uShowNodata: { value: 1 },
      uTime: { value: 0 }, uSun: { value: sunDir },
      uNew: { value: new THREE.Color("#1f5fd6") }, uOld: { value: new THREE.Color("#8fbbe3") },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv; varying vec3 vW;
      void main() { vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D uA, uB, uPre; uniform float uT, uHasPre, uOn, uShowNew, uShowPre, uShowNodata, uTime;
      uniform vec3 uSun, uNew, uOld;
      varying vec2 vUv; varying vec3 vW;
      void main() {
        vec4 a = texture2D(uA, vUv), b = texture2D(uB, vUv);
        float p = mix(a.r, b.r, uT), nd = mix(a.g, b.g, uT);
        float w = smoothstep(0.47, 0.53, p);            // textures hold probability: draw the p = 0.5 contour
        float before = uHasPre * smoothstep(0.47, 0.53, texture2D(uPre, vUv).r);
        float aw = 0.9 * uOn * w * mix(uShowNew, uShowPre, before);
        float hatch = step(0.5, fract((vW.x + vW.z) * 4.0));
        float an = uOn * uShowNodata * smoothstep(0.4, 0.6, nd) * (0.12 + 0.3 * hatch) * (1.0 - aw);
        float alpha = aw + an;
        if (alpha < 0.01) discard;
        vec3 col = mix(uNew, uOld, before);
        float ripple = 0.04 * sin(vW.x * 9.0 + uTime * 1.3) * sin(vW.z * 7.0 - uTime * 1.1);
        vec3 v = normalize(cameraPosition - vW);
        float spec = pow(max(dot(reflect(-uSun, vec3(0.0, 1.0, 0.0)), v), 0.0), 40.0);
        col = col * (0.95 + ripple) + vec3(0.9, 0.95, 1.0) * spec * 0.35;
        vec3 c = (col * aw + vec3(0.36, 0.37, 0.42) * an) / alpha;
        gl_FragColor = vec4(c, alpha);
        #include <colorspace_fragment>
      }`,
  });
  const water = new THREE.Mesh(geo, waterMat);
  water.renderOrder = 1;
  scene.add(water);

  // ---------------------------------------------------------------- cross-fades
  const fades: { u: { value: number }; t0: number; ms: number }[] = [];
  const fade = (u: { value: number }, ms: number) => { u.value = 0; fades.push({ u, t0: performance.now(), ms }); };
  let shownPhoto = "", shownWater = "", reqPhoto = 0, reqWater = 0;

  const setPhoto = async (url: string) => {
    if (url === shownPhoto) return;
    const id = ++reqPhoto;
    inUse.add(url);
    let t: THREE.Texture;
    try {
      t = await tex(url, true);
    } catch {
      texCache.delete(url);
      t = await tex(url.replace("/composite/", "/frames/"), true);
    }
    if (id !== reqPhoto) return;
    inUse.delete(shownPhoto);
    shownPhoto = url;
    const u = groundMat.uniforms;
    u.uA.value = u.uT.value >= 0.5 ? u.uB.value : u.uA.value;
    u.uB.value = t;
    fade(u.uT, st.playing ? 350 : 600);
    drawMiniSoon();
  };
  const setWater = async (a: Analysis | null) => {
    const url = a ? waterUrl(a) : "";
    if (url === shownWater) return;
    const id = ++reqWater;
    const u = waterMat.uniforms;
    if (!a) { shownWater = ""; u.uOn.value = 0; drawMiniSoon(); return; }
    inUse.add(url);
    const t = await tex(url, false);
    if (id !== reqWater) return;
    inUse.delete(shownWater);
    const wasOn = u.uOn.value > 0;
    shownWater = url;
    u.uA.value = wasOn ? (u.uT.value >= 0.5 ? u.uB.value : u.uA.value) : t;
    u.uB.value = t;
    u.uOn.value = 1;
    if (wasOn) fade(u.uT, st.playing ? 350 : 700); else u.uT.value = 1;
    drawMiniSoon();
  };
  const setPre = async () => {
    const pre = meta.pre_event[st.year];
    const u = waterMat.uniforms;
    if (!pre) { u.uHasPre.value = 0; u.uPre.value = blank; return; }
    const url = `${BASE}water_${pre}.png`;
    inUse.add(url);
    u.uPre.value = await tex(url, false);
    u.uHasPre.value = 1;
  };

  // ---------------------------------------------------------------- CPU copies for the cursor and minimap
  const M = 440;
  const maskCache = new Map<string, Promise<{ n: number; r: Uint8Array; g: Uint8Array; layer: OffscreenCanvas }>>();
  const maskOf = (url: string) => {
    let p = maskCache.get(url);
    if (!p) {
      p = (async () => {
        const n = 1024;
        const bmp = await createImageBitmap(await (await fetch(url)).blob(), { resizeWidth: n, resizeHeight: n, resizeQuality: "medium" });
        const c = new OffscreenCanvas(n, n);
        const ctx = c.getContext("2d")!;
        ctx.drawImage(bmp, 0, 0);
        const d = ctx.getImageData(0, 0, n, n).data;
        const r = new Uint8Array(n * n), g = new Uint8Array(n * n);
        for (let i = 0; i < n * n; i++) { r[i] = d[i * 4]; g[i] = d[i * 4 + 1]; }
        // Minimap overlay at minimap resolution.
        const layer = new OffscreenCanvas(M, M);
        const img = new ImageData(M, M);
        for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) {
          const k = Math.floor((y / M) * n) * n + Math.floor((x / M) * n), o = (y * M + x) * 4;
          if (r[k] > 127) { img.data[o] = 31; img.data[o + 1] = 95; img.data[o + 2] = 214; img.data[o + 3] = 235; }
          else if (g[k] > 127) { img.data[o] = 92; img.data[o + 1] = 94; img.data[o + 2] = 107; img.data[o + 3] = 110; }
        }
        layer.getContext("2d")!.putImageData(img, 0, 0);
        return { n, r, g, layer };
      })();
      maskCache.set(url, p);
    }
    return p;
  };
  const bmpCache = new Map<string, Promise<ImageBitmap>>();
  const miniPhoto = (url: string) => {
    let p = bmpCache.get(url);
    if (!p) {
      p = fetch(url).then((r) => r.blob()).then((b) => createImageBitmap(b, { resizeWidth: M, resizeHeight: M, resizeQuality: "medium" }));
      bmpCache.set(url, p);
      if (bmpCache.size > 40) bmpCache.delete(bmpCache.keys().next().value!);
    }
    return p;
  };

  // ---------------------------------------------------------------- minimap
  const mini = $<HTMLCanvasElement>("mini");
  const mctx = mini.getContext("2d")!;
  const toMini = (x: number, z: number): [number, number] => [((x + W / 2) / W) * M, ((z + H / 2) / H) * M];
  const ray = new THREE.Raycaster();
  const plane0 = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const corner = new THREE.Vector3();
  const footprint = (): [number, number][] => [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([cx, cy]) => {
    ray.setFromCamera(new THREE.Vector2(cx, cy), camera);
    if (ray.ray.intersectPlane(plane0, corner) && corner.distanceTo(camera.position) < 300) return toMini(corner.x, corner.z);
    const d = ray.ray.direction.clone().setY(0).normalize().multiplyScalar(200).add(camera.position);
    return toMini(d.x, d.z);
  });
  let miniBg: ImageBitmap | null = null, miniWater: OffscreenCanvas | null = null, miniDirty = true;
  const drawMiniSoon = () => {
    const ph = shownPhoto, wa = shownWater;
    if (ph) miniPhoto(ph).then((b) => { if (ph === shownPhoto) { miniBg = b; miniDirty = true; } });
    if (wa) maskOf(wa).then((m) => { if (wa === shownWater) { miniWater = m.layer; miniDirty = true; } });
    else { miniWater = null; miniDirty = true; }
  };
  const drawMini = () => {
    mctx.fillStyle = "#ddd4c6";
    mctx.fillRect(0, 0, M, M);
    if (miniBg && st.ground !== "relief") mctx.drawImage(miniBg, 0, 0);
    if (miniWater && waterMat.uniforms.uOn.value) mctx.drawImage(miniWater, 0, 0);
    mctx.save();
    mctx.beginPath(); mctx.rect(0, 0, M, M); mctx.clip();
    const fp = footprint();
    mctx.beginPath();
    fp.forEach(([x, y], k) => (k ? mctx.lineTo(x, y) : mctx.moveTo(x, y)));
    mctx.closePath();
    mctx.fillStyle = "rgba(255,255,255,0.18)"; mctx.fill();
    mctx.lineWidth = 4; mctx.strokeStyle = "#c8553d"; mctx.stroke();
    mctx.restore();
    const [px, py] = toMini(camera.position.x, camera.position.z);
    const [tx, ty] = toMini(controls.target.x, controls.target.z);
    mctx.save();
    mctx.translate(Math.min(M - 10, Math.max(10, px)), Math.min(M - 10, Math.max(10, py)));
    mctx.rotate(Math.atan2(ty - py, tx - px));
    mctx.beginPath(); mctx.moveTo(14, 0); mctx.lineTo(-8, -9); mctx.lineTo(-4, 0); mctx.lineTo(-8, 9); mctx.closePath();
    mctx.fillStyle = "#2a2520"; mctx.strokeStyle = "#fff"; mctx.lineWidth = 2; mctx.fill(); mctx.stroke();
    mctx.restore();
    mctx.beginPath(); mctx.arc(tx, ty, 5, 0, Math.PI * 2); mctx.fillStyle = "#c8553d"; mctx.fill();
  };
  mini.addEventListener("click", (e) => {
    const r = mini.getBoundingClientRect();
    const dx = ((e.clientX - r.left) / r.width) * W - W / 2 - controls.target.x;
    const dz = ((e.clientY - r.top) / r.height) * H - H / 2 - controls.target.z;
    controls.target.x += dx; controls.target.z += dz;
    camera.position.x += dx; camera.position.z += dz;
  });

  // ---------------------------------------------------------------- charts (SVG)
  const weatherOf = () => weather.years[st.year];
  const wxAt = (day: number) => {
    const w = weatherOf(), i = w.days.indexOf(toIso(day));
    return i < 0 ? { rain: null, q: null } : { rain: w.precip_mm[i], q: w.discharge_m3s[i] };
  };
  const rain7 = (day: number) => {
    let s = 0, n = 0;
    for (let k = 0; k < 7; k++) { const v = wxAt(day - k).rain; if (v != null) { s += v; n++; } }
    return n ? s : null;
  };

  const wxBox = $("wx-chart");
  const drawWeather = () => {
    const w = weatherOf(), [d0, d1] = range();
    const Wd = 272, Hh = 128, top = 8, bot = 18;
    const x = (d: number) => ((d - d0) / (d1 - d0)) * Wd;
    const rMax = Math.max(10, ...w.precip_mm.map((v) => v ?? 0));
    const qMax = Math.max(1, ...w.discharge_m3s.map((v) => v ?? 0));
    const yR = (v: number) => Hh - bot - (v / rMax) * (Hh - top - bot);
    const yQ = (v: number) => Hh - bot - (v / qMax) * (Hh - top - bot);
    let s = `<svg viewBox="0 0 ${Wd} ${Hh}" role="img" aria-label="Lluvia diaria y caudal del río Piura">`;
    for (const e of keyEvents()) if (e.kind === "hazard") s += `<line x1="${x(toDay(e.date))}" x2="${x(toDay(e.date))}" y1="${top}" y2="${Hh - bot}" stroke="rgba(200,114,42,0.45)" stroke-dasharray="2 3"/>`;
    w.days.forEach((d, i) => {
      const v = w.precip_mm[i];
      if (!v) return;
      const xx = x(toDay(d));
      s += `<rect x="${xx - 0.8}" y="${yR(v)}" width="1.6" height="${Hh - bot - yR(v)}" fill="#5b8fd0"/>`;
    });
    const pts = w.days.map((d, i) => (w.discharge_m3s[i] == null ? null : `${x(toDay(d)).toFixed(1)},${yQ(w.discharge_m3s[i]!).toFixed(1)}`)).filter(Boolean);
    s += `<polyline points="${pts.join(" ")}" fill="none" stroke="#c8722a" stroke-width="1.6"/>`;
    for (let m = 0; m < 5; m++) {
      const dm = toDay(`${st.year}-0${m + 1}-01`);
      s += `<line x1="${x(dm)}" x2="${x(dm)}" y1="${Hh - bot}" y2="${Hh - bot + 4}" stroke="#b9ad9c"/><text x="${x(dm) + 3}" y="${Hh - 4}">${MONTHS[m]}</text>`;
    }
    s += `<line x1="0" x2="${Wd}" y1="${Hh - bot}" y2="${Hh - bot}" stroke="#b9ad9c"/>`;
    s += `<text x="0" y="${top}" style="fill:#3f74b8">lluvia, máx ${rMax.toFixed(0)} mm/día</text>`;
    s += `<text x="${Wd}" y="${top}" text-anchor="end" style="fill:#b5641f">caudal, máx ${qMax.toFixed(0)} m³/s</text>`;
    s += `<line id="wx-head" x1="0" x2="0" y1="${top}" y2="${Hh - bot}" stroke="#2a2520" stroke-width="1.2"/></svg>`;
    wxBox.innerHTML = s;
    $("wx-foot").innerHTML = `Lluvia: ${esc(weather.precip.note)}. Caudal: ${esc(weather.discharge.note)}. Modelos (ERA5, GloFAS), no mediciones.`;
  };

  // ---------------------------------------------------------------- timeline track
  const track = $("track");
  let trackW = 0;
  const TH = 62, AX = 42;
  const tx = (d: number) => { const [d0, d1] = range(); return ((d - d0) / (d1 - d0)) * trackW; };
  const dayFromX = (px: number) => { const [d0, d1] = range(); return Math.round(d0 + Math.min(1, Math.max(0, px / trackW)) * (d1 - d0)); };
  const drawTrack = () => {
    trackW = track.clientWidth;
    const [d0, d1] = range(), w = weatherOf();
    const rMax = Math.max(10, ...w.precip_mm.map((v) => v ?? 0));
    let s = `<svg viewBox="0 0 ${trackW} ${TH}" aria-hidden="true">`;
    // Hazard period (first to last hazard event).
    const hz = keyEvents().filter((e) => e.kind === "hazard").map((e) => toDay(e.date));
    if (hz.length > 1) s += `<rect x="${tx(Math.min(...hz))}" y="6" width="${tx(Math.max(...hz)) - tx(Math.min(...hz))}" height="${AX - 6}" fill="rgba(200,114,42,0.1)"/>`;
    w.days.forEach((d, i) => {
      const v = w.precip_mm[i];
      if (!v) return;
      const h = (v / rMax) * 16;
      s += `<rect x="${tx(toDay(d)) - 0.7}" y="${AX - h}" width="1.4" height="${h}" fill="rgba(63,116,184,0.45)"/>`;
    });
    s += `<line x1="0" x2="${trackW}" y1="${AX}" y2="${AX}" stroke="#b9ad9c" stroke-width="2" stroke-linecap="round"/>`;
    for (let d = d0; d <= d1; d++) {
      const t = new Date(d * DAY);
      if (t.getUTCDate() === 1) s += `<line x1="${tx(d)}" x2="${tx(d)}" y1="${AX}" y2="${AX + 6}" stroke="#a99d8c"/><text x="${tx(d) + 3}" y="${TH - 2}">${MONTHS_LONG[t.getUTCMonth()]}</text>`;
      else if (t.getUTCDay() === 1) s += `<line x1="${tx(d)}" x2="${tx(d)}" y1="${AX}" y2="${AX + 3}" stroke="#cfc5b6"/>`;
    }
    for (const f of framesOf()) {
      const c = f.valid >= 0.9 && f.cloud <= 0.35 ? "#2a2520" : f.valid >= 0.9 ? "#a99d8c" : "#d6cdbf";
      s += `<rect x="${tx(toDay(f.date)) - 1.5}" y="${AX + 5}" width="3" height="7" rx="1" fill="${c}"><title>${fmtDay(toDay(f.date))}</title></rect>`;
    }
    for (const a of analysesOf()) {
      const xx = tx(toDay(a.date));
      s += `<path d="M${xx} ${AX - 13} l5 5 l-5 5 l-5 -5z" fill="${a.comparable ? "#1f5fd6" : "#fff"}" stroke="#1f5fd6" stroke-width="1.5"/>`;
    }
    keyEvents().forEach((e, i) => {
      const xx = tx(toDay(e.date));
      s += `<g class="ev-hit" data-ev="${i}"><line x1="${xx}" x2="${xx}" y1="4" y2="${AX}" stroke="#c8722a" stroke-width="1.2"/><path d="M${xx} 3 h9 l-3 3.5 l3 3.5 h-9z" fill="#c8722a"/><rect x="${xx - 6}" y="0" width="18" height="${AX}" fill="transparent"/></g>`;
    });
    s += `<g id="head"><line x1="0" x2="0" y1="0" y2="${AX + 12}" stroke="#2a2520" stroke-width="1.5"/><circle cx="0" cy="${AX}" r="6" fill="#fff" stroke="#1f5fd6" stroke-width="2.5"/></g></svg>`;
    track.innerHTML = s;
    track.querySelectorAll<SVGGElement>(".ev-hit").forEach((g) => {
      const e = keyEvents()[Number(g.dataset.ev)];
      g.addEventListener("pointerenter", (ev) => showTip(ev, e));
      g.addEventListener("pointerleave", hideTip);
      g.addEventListener("pointerdown", (ev) => { ev.stopPropagation(); setDay(toDay(e.date)); });
    });
    moveHead();
  };
  const tipEl = $("tip");
  const showTip = (ev: PointerEvent, e: KeyEvent) => {
    tipEl.innerHTML = `<b>${fmtDay(toDay(e.date))}</b> · ${esc(e.label)}<br><span class="muted small">${esc(e.source)}</span>`;
    const r = (ev.currentTarget as Element).getBoundingClientRect();
    tipEl.style.left = `${r.left + 6}px`;
    tipEl.style.top = `${r.top}px`;
    tipEl.hidden = false;
  };
  const hideTip = () => (tipEl.hidden = true);
  const moveHead = () => {
    const h = track.querySelector<SVGGElement>("#head");
    if (h) h.setAttribute("transform", `translate(${tx(st.day)},0)`);
    const [d0, d1] = range();
    const wh = wxBox.querySelector("#wx-head");
    if (wh) { const xx = ((st.day - d0) / (d1 - d0)) * 272; wh.setAttribute("x1", `${xx}`); wh.setAttribute("x2", `${xx}`); }
  };
  let scrubbing = false;
  track.addEventListener("pointerdown", (e) => {
    scrubbing = true;
    track.setPointerCapture(e.pointerId);
    setDay(dayFromX(e.clientX - track.getBoundingClientRect().left));
  });
  track.addEventListener("pointermove", (e) => { if (scrubbing) setDay(dayFromX(e.clientX - track.getBoundingClientRect().left)); });
  track.addEventListener("pointerup", () => (scrubbing = false));

  // ---------------------------------------------------------------- calendar
  const cal = $("cal"), calBtn = $("cal-btn");
  const drawCal = () => {
    const [d0, d1] = range();
    const photos = new Map(framesOf().map((f) => [f.date, f]));
    const ai = new Set(analysesOf().map((a) => a.date));
    const evs = new Map(keyEvents().map((e) => [e.date, e]));
    let s = `<div style="display:flex;flex-wrap:wrap;gap:16px;max-width:660px">`;
    for (let m = new Date(d0 * DAY).getUTCMonth(); m <= new Date(d1 * DAY).getUTCMonth(); m++) {
      const first = toDay(`${st.year}-${String(m + 1).padStart(2, "0")}-01`);
      const lead = new Date(first * DAY).getUTCDay();
      s += `<div class="month"><h4>${MONTHS_LONG[m]} ${st.year}</h4><div class="grid">${["d", "l", "m", "m", "j", "v", "s"].map((x) => `<span class="dow">${x}</span>`).join("")}`;
      s += "<span></span>".repeat(lead);
      for (let d = first; new Date(d * DAY).getUTCMonth() === m; d++) {
        const iso = toIso(d), f = photos.get(iso), e = evs.get(iso);
        const cls = ["day", f ? "photo" : "", f && f.valid >= 0.9 && f.cloud <= 0.35 ? "clear" : "", ai.has(iso) ? "ai" : "", e ? "ev" : "", d === st.day ? "cur" : ""].join(" ");
        const title = [f ? `foto (nubes ${pct(f.cloud)}, cobertura ${pct(f.valid)})` : "", ai.has(iso) ? "análisis IA" : "", e ? e.label : ""].filter(Boolean).join(" · ");
        s += `<button class="${cls}" data-d="${d}" title="${esc(title)}">${new Date(d * DAY).getUTCDate()}</button>`;
      }
      s += `</div></div>`;
    }
    s += `</div><div class="legend"><b style="color:var(--ink)">Leyenda</b><span>● oscuro: foto despejada</span><span>● claro: foto con nubes o parcial</span><span><span style="box-shadow:inset 0 0 0 1.5px var(--new);border-radius:4px;padding:0 4px">n</span> análisis de agua IA</span><span style="color:var(--accent)">n evento clave</span></div>`;
    cal.innerHTML = s;
    cal.querySelectorAll<HTMLButtonElement>(".day").forEach((b) => (b.onclick = () => { setDay(Number(b.dataset.d)); cal.hidden = true; calBtn.setAttribute("aria-expanded", "false"); }));
  };
  calBtn.onclick = () => {
    const open = cal.hidden;
    if (open) drawCal();
    cal.hidden = !open;
    calBtn.setAttribute("aria-expanded", String(open));
  };
  document.addEventListener("pointerdown", (e) => { if (!cal.hidden && !cal.contains(e.target as Node) && !calBtn.contains(e.target as Node)) { cal.hidden = true; calBtn.setAttribute("aria-expanded", "false"); } });

  // ---------------------------------------------------------------- panels
  const renderIncident = () => {
    const ev = yr().event;
    $("event-kicker").textContent = yr().kicker;
    $("event-title").textContent = ev.name;
    const period = ev.facts.find((f) => f.label.startsWith("Fechas"));
    $("event-sub").textContent = `${ev.location}${period ? ` · ${period.value}` : ""}`;
  };
  const renderAnalysis = () => {
    const as = analysesOf(), cur = analysisAt(st.day);
    const maxA = Math.max(1, ...as.map((a) => a.flooded_km2 ?? a.water_km2));
    const [d0, d1] = [Math.min(...as.map((a) => toDay(a.date))) - 3, Math.max(...as.map((a) => toDay(a.date))) + 3];
    const Wd = 300, Hh = 120, bot = 18;
    const x = (d: number) => ((d - d0) / Math.max(1, d1 - d0)) * (Wd - 20) + 10;
    let chart = `<svg viewBox="0 0 ${Wd} ${Hh}" role="img" aria-label="Área inundada por fecha">`;
    chart += `<line x1="0" x2="${Wd}" y1="${Hh - bot}" y2="${Hh - bot}" stroke="#b9ad9c"/>`;
    for (const a of as) {
      const v = a.is_pre ? a.water_km2 : a.flooded_km2 ?? a.water_km2, h = (v / maxA) * (Hh - bot - 16), xx = x(toDay(a.date));
      const fill = a.is_pre ? "#8fbbe3" : a.comparable ? "#1f5fd6" : "transparent";
      chart += `<rect x="${xx - 7}" y="${Hh - bot - h}" width="14" height="${Math.max(1, h)}" rx="2" fill="${fill}" stroke="${a.is_pre ? "#8fbbe3" : "#1f5fd6"}" stroke-width="${cur?.date === a.date ? 2.5 : 1.2}"/>`;
      chart += `<text x="${xx}" y="${Hh - bot - h - 4}" text-anchor="middle" style="fill:#2a2520">${v.toFixed(0)}</text>`;
      chart += `<text x="${xx}" y="${Hh - 5}" text-anchor="middle">${a.label.split(" ").slice(0, 2).join(" ")}</text>`;
    }
    chart += `</svg>`;
    const rows = as.map((a) => `<tr class="${cur?.date === a.date ? "cur" : a.comparable ? "" : "dim"}" data-d="${toDay(a.date)}" style="cursor:pointer">
      <td>${a.label.replace(/ \d{4}$/, "")}${a.is_pre ? " ·pre" : ""}</td><td>${a.water_km2.toFixed(1)}</td><td>${a.flooded_km2 != null ? a.flooded_km2.toFixed(1) : "—"}</td><td>${pct(a.nodata_share)}</td></tr>`).join("");
    const spreads = as.filter((a) => a.spread).map((a) => {
      const s = a.spread!;
      return `<tr><td>${fmtDay(toDay(s.from), false)} → ${fmtDay(toDay(a.date), false)}</td><td>+${s.new_km2.toFixed(1)}</td><td>−${s.receded_km2.toFixed(1)}</td><td class="${s.net_km2_per_day > 0 ? "up" : "down"}">${s.net_km2_per_day > 0 ? "+" : ""}${s.net_km2_per_day.toFixed(1)}</td></tr>`;
    }).join("");
    $("tab-analysis").innerHTML = `
      <div class="group"><div class="glabel">${meta.pre_event[st.year] ? "Área inundada por fecha (km²)" : "Agua detectada por fecha (km²)"}</div>
        <div class="chart">${chart}</div>
        <div class="src">Barras llenas: fechas comparables (≤ 20 % de la zona sin dato). Vacías: fechas parciales (borde de pasada o nubes), cifras incompletas.${meta.pre_event[st.year] ? " Celeste: imagen pre-evento (agua previa)." : ""}</div></div>
      <div class="group"><div class="glabel">Fechas analizadas</div>
        <table class="t"><thead><tr><th>Fecha</th><th>Agua</th><th>Inundada</th><th>Sin dato</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${spreads ? `<div class="group"><div class="glabel">Velocidad de avance</div>
        <table class="t"><thead><tr><th>Tramo</th><th>Nueva</th><th>Retirada</th><th>km²/día</th></tr></thead><tbody>${spreads}</tbody></table>
        <div class="src">Solo píxeles despejados en ambas fechas y en la pre-evento (~82 % de la zona). Neto = (inundada nueva − retirada) / días.</div></div>` : ""}
      <div class="note">${meta.pre_event[st.year]
        ? `<b>Área inundada</b> = agua de la IA que no estaba en la imagen pre-evento (${fmtDay(toDay(meta.pre_event[st.year]!))}), solo donde ambas fechas están despejadas. Incluye todo el cuadro de 50 × 50 km.`
        : `<b>Sin área inundada para 2017:</b> falta una fecha de inicio del evento con fuente para elegir la imagen pre-evento. Se muestra el agua total detectada.`}</div>`;
    $("tab-analysis").querySelectorAll<HTMLTableRowElement>("tr[data-d]").forEach((r) => (r.onclick = () => setDay(Number(r.dataset.d))));
  };
  const renderEvidence = () => {
    const y = yr(), refs = analysesOf().filter((a) => a.reference);
    const refHtml = refs.map((a) => `<div class="group"><div class="glabel">${a.label} · ${esc(a.reference!.source)}</div>
      <div class="metric3"><div><span class="lbl">IoU</span><span class="v">${a.reference!.iou.toFixed(3)}</span></div><div><span class="lbl">Precisión</span><span class="v">${a.reference!.precision.toFixed(3)}</span></div><div><span class="lbl">Exhaustividad</span><span class="v">${a.reference!.recall.toFixed(3)}</span></div></div></div>`).join("");
    $("tab-evidence").innerHTML = `
      <div class="group"><div class="glabel">Precisión del modelo (prueba, una sola vez)</div>
        <div class="metric3"><div><span class="lbl">IoU</span><span class="v">${y.test.iou.toFixed(3)}</span></div><div><span class="lbl">Precisión</span><span class="v">${y.test.precision.toFixed(3)}</span></div><div><span class="lbl">Exhaustividad</span><span class="v">${y.test.recall.toFixed(3)}</span></div></div>
        <div class="src">${y.test.chips} recortes de ${y.test.set}, media de 3 semillas (evaluación congelada).</div></div>
      ${refHtml ? `<div class="glabel" style="margin-bottom:-6px">Concordancia en esta zona</div>${refHtml}<div class="src">Zona de 50 × 50 km, píxeles válidos en ambos. Gran parte de la zona fue entrenamiento en 2023: cifra optimista.</div>` : ""}
      <div class="group"><div class="glabel">Datos del evento</div>
        <dl class="facts">${y.event.facts.map((f) => `<dt title="${esc(f.source)}">${esc(f.label)}</dt><dd title="${esc(f.source)}">${esc(f.value)}</dd>`).join("")}</dl>
        <div class="src">${[...new Set(y.event.facts.map((f) => f.source))].map(esc).join(" · ")}</div></div>
      <div class="note warn"><b>Tenga en cuenta.</b> Prototipo de investigación, no un servicio oficial. Bajo las nubes y fuera de la pasada del satélite no hay resultado. La foto «sin nubes» mezcla fechas píxel a píxel. Lluvia y caudal vienen de modelos (ERA5, GloFAS), no de estaciones.</div>
      <div class="src">Relieve: Copernicus DEM GLO-30 (modelo de superficie). Imágenes: Sentinel-2 L1C (CDSE). Agua: U-Net entrenada con Perú, 3 semillas, p &gt; 0,5.</div>`;
  };

  const renderKpis = () => {
    const a = comparableAt(st.day), shown = analysisAt(st.day), hasPre = !!meta.pre_event[st.year];
    if (hasPre && a?.flooded_km2 != null) {
      $("k-flood").textContent = km2(a.flooded_km2);
      $("k-flood-sub").textContent = `${a.label} frente a ${fmtDay(toDay(meta.pre_event[st.year]!))} · ${pct(a.flood_observed_share ?? 0)} de la zona observada en ambas`;
    } else {
      $("k-flood").textContent = "—";
      $("k-flood-sub").textContent = hasPre ? "aún no hay una fecha comparable después del inicio" : "2017: falta la fecha de inicio del evento para la imagen pre-evento";
    }
    if (a?.spread) {
      const s = a.spread, up = s.net_km2_per_day > 0;
      $("k-rate").innerHTML = `<span class="${up ? "up" : "down"}">${up ? "▲ +" : "▼ −"}${Math.abs(s.net_km2_per_day).toFixed(1)}</span><span style="font-size:11px;color:var(--ink-2)"> km²/día</span>`;
      $("k-rate-sub").textContent = `${fmtDay(toDay(s.from), false)} → ${fmtDay(toDay(a.date), false)} · +${s.new_km2.toFixed(0)} / −${s.receded_km2.toFixed(0)} km²`;
    } else {
      $("k-rate").textContent = "—";
      $("k-rate-sub").textContent = "requiere dos fechas comparables";
    }
    const w = shown ?? a;
    $("k-water").textContent = w ? km2(w.water_km2) : "—";
    $("k-water-sub").textContent = w ? `${w.label}${w.comparable ? "" : " · parcial"}` : "sin análisis aún";
    $("k-obs").textContent = w ? pct(1 - w.nodata_share) : "—";
    $("k-obs-sub").textContent = w ? `imagen del ${w.label}` : "con resultado de la IA";
    const r = rain7(st.day);
    $("k-rain").textContent = r == null ? "—" : `${r.toFixed(0)} mm`;
    const wx = wxAt(st.day);
    $("w-rain").textContent = wx.rain == null ? "—" : `${wx.rain.toFixed(1)} mm`;
    $("w-q").textContent = wx.q == null ? "—" : `${wx.q.toFixed(0)} m³/s`;
  };

  const groundNote = () => {
    const f = photoAt(st.day);
    if (st.ground === "relief") return "Color por altitud (0–101 m).";
    if (!f) return "Sin foto antes de esta fecha.";
    return st.ground === "composite"
      ? `Imagen sin nubes al ${fmtDay(toDay(f.date))}: cada píxel muestra la mediana de sus 3 últimas observaciones despejadas, así que los cambios rápidos aparecen con 1–2 pasadas de retraso (el agua de la IA sí es de su fecha). Despejado en esta pasada: ${pct(f.fresh_share)} de la zona.`
      : `Pasada del ${fmtDay(toDay(f.date))}: nubes ${pct(f.cloud)}, cobertura ${pct(f.valid)}.`;
  };

  // ---------------------------------------------------------------- day changes
  let lastAnalysisDate = "";
  const setDay = (d: number) => {
    const [d0, d1] = range();
    st.day = Math.min(d1, Math.max(d0, d));
    const t = new Date(st.day * DAY);
    $("p-date").textContent = `${DOW[t.getUTCDay()]} ${fmtDay(st.day)}`;
    $("cal-label").textContent = fmtDay(st.day);
    const f = photoAt(st.day), a = analysisAt(st.day);
    $("p-info").textContent = `${f ? `${st.ground === "raw" ? "Foto" : "Imagen"}: ${fmtDay(toDay(f.date), false)}` : "Sin foto"} · ${a ? `Agua IA: ${a.label.replace(/ \d{4}$/, "")}` : "sin análisis IA"}`;
    $("mini-date").textContent = f ? fmtDay(toDay(f.date)) : "";
    $("ground-note").textContent = groundNote();
    if (st.ground !== "relief" && f) setPhoto(photoUrl(f));
    setWater(a);
    if (st.playing) prefetch();
    renderKpis();
    if ((a?.date ?? "") !== lastAnalysisDate) { lastAnalysisDate = a?.date ?? ""; renderAnalysis(); }
    moveHead();
  };
  // While playing, decode and upload the next photo and water textures before they are shown.
  const prefetched = new Set<string>();
  const prefetch = () => {
    const nf = framesOf().find((f) => toDay(f.date) > st.day && (st.ground !== "raw" || usableRaw(f)));
    const na = analysesOf().find((a) => toDay(a.date) > st.day && (a.comparable || st.cloudyAI));
    for (const [url, srgb] of [[nf && st.ground !== "relief" ? photoUrl(nf) : "", true], [na ? waterUrl(na) : "", false]] as const) {
      if (!url || prefetched.has(url)) continue;
      prefetched.add(url);
      tex(url, srgb).then((t) => renderer.initTexture(t)).catch(() => prefetched.delete(url));
    }
  };
  const setYear = async (y: string) => {
    st.year = y;
    $("event-seg").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.y === y));
    renderIncident();
    const hasPre = !!meta.pre_event[y];
    $("l-new-label").textContent = hasPre ? "Agua nueva (inundación)" : "Agua detectada (sin imagen pre-evento)";
    $("l-pre-row").hidden = !hasPre;
    drawWeather();
    drawTrack();
    await setPre();
    lastAnalysisDate = "-";
    renderEvidence();
    setDay(toDay(yr().initial));
  };

  // ---------------------------------------------------------------- controls
  const seg = $("event-seg");
  for (const y of ["2017", "2023"]) {
    const b = document.createElement("button");
    b.dataset.y = y;
    b.textContent = YEARS[y].event.short;
    b.onclick = () => setYear(y);
    seg.appendChild(b);
  }
  const step = (dir: 1 | -1) => {
    const ds = [...new Set([...framesOf().filter((f) => st.ground !== "raw" || usableRaw(f)).map((f) => toDay(f.date)), ...analysesOf().map((a) => toDay(a.date))])].sort((a, b) => a - b);
    const next = dir > 0 ? ds.find((d) => d > st.day) : [...ds].reverse().find((d) => d < st.day);
    if (next != null) setDay(next);
  };
  $("p-prev").onclick = () => step(-1);
  $("p-next").onclick = () => step(1);
  const playBtn = $("p-play");
  let acc = 0;
  const setPlaying = (on: boolean) => {
    st.playing = on;
    playBtn.textContent = on ? "❚❚" : "▶";
    playBtn.setAttribute("aria-label", on ? "Pausa" : "Reproducir");
    if (on && st.day >= range()[1]) setDay(range()[0]);
    acc = 0;
  };
  playBtn.onclick = () => setPlaying(!st.playing);
  $<HTMLSelectElement>("p-speed").onchange = (e) => (st.speed = Number((e.target as HTMLSelectElement).value));
  document.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(!st.playing); }
    if (e.key === "ArrowRight") setDay(st.day + (e.shiftKey ? 7 : 1));
    if (e.key === "ArrowLeft") setDay(st.day - (e.shiftKey ? 7 : 1));
  });

  const tabs = document.querySelectorAll<HTMLButtonElement>(".tabs button");
  tabs.forEach((b) => (b.onclick = () => {
    tabs.forEach((x) => x.classList.toggle("on", x === b));
    for (const id of ["layers", "analysis", "evidence"]) $(`tab-${id}`).hidden = id !== b.dataset.tab;
  }));
  const bindToggle = (id: string, f: (on: boolean) => void) => { const el = $<HTMLInputElement>(id); el.onchange = () => f(el.checked); };
  bindToggle("l-new", (on) => (waterMat.uniforms.uShowNew.value = on ? 1 : 0));
  bindToggle("l-pre", (on) => (waterMat.uniforms.uShowPre.value = on ? 1 : 0));
  bindToggle("l-nodata", (on) => (waterMat.uniforms.uShowNodata.value = on ? 1 : 0));
  bindToggle("l-cloudy", (on) => { st.cloudyAI = on; lastAnalysisDate = "-"; setDay(st.day); });
  bindToggle("l-allphotos", (on) => { st.rawAll = on; setDay(st.day); });
  $("ground-seg").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => {
    st.ground = b.dataset.ground as typeof st.ground;
    $("ground-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    groundMat.uniforms.uRelief.value = st.ground === "relief" ? 1 : 0;
    shownPhoto = "";
    setDay(st.day);
  }));
  $<HTMLInputElement>("exag").oninput = (e) => { exag = Number((e.target as HTMLInputElement).value); applyHeights(); };

  // View: 3D orbit or top-down, with a short camera move.
  let camMove: { from: THREE.Vector3; to: THREE.Vector3; t0: number } | null = null;
  $("view-seg").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => {
    $("view-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    const tg = controls.target, dist = camera.position.distanceTo(tg);
    const to = b.dataset.view === "top" ? new THREE.Vector3(tg.x, tg.y + dist, tg.z + 0.01) : new THREE.Vector3(tg.x + dist * 0.08, tg.y + dist * 0.6, tg.z + dist * 0.79);
    camMove = { from: camera.position.clone(), to, t0: performance.now() };
  }));

  // ---------------------------------------------------------------- cursor readout (height-field picking)
  const surface = (x: number, z: number) => {
    const fx = Math.min(G - 1.001, Math.max(0, ((x + W / 2) / W) * (G - 1)));
    const fz = Math.min(G - 1.001, Math.max(0, ((z + H / 2) / H) * (G - 1)));
    const c = Math.floor(fx), r = Math.floor(fz), a = fx - c, b = fz - r;
    const h = heights[r * G + c] * (1 - a) * (1 - b) + heights[r * G + c + 1] * a * (1 - b)
      + heights[(r + 1) * G + c] * (1 - a) * b + heights[(r + 1) * G + c + 1] * a * b;
    return (h / 1000) * exag;
  };
  const topY = (meta.height_max / 1000) * 80 + 0.01;
  const p = new THREE.Vector3();
  const pick = (): { u: number; v: number } | null => {
    const o = ray.ray.origin, d = ray.ray.direction;
    if (d.y >= 0 && o.y > topY) return null;
    let t = o.y > topY ? (topY - o.y) / d.y : 0;
    const stp = W / G / 2, tMax = t + 300;
    for (; t < tMax; t += stp) {
      ray.ray.at(t, p);
      if (p.y < 0) break;
      if (Math.abs(p.x) > W / 2 + stp || Math.abs(p.z) > H / 2 + stp) continue;
      if (p.y <= surface(p.x, p.z)) break;
    }
    if (t >= tMax) return null;
    let lo = t - stp, hi = t;
    for (let k = 0; k < 10; k++) {
      const mid = (lo + hi) / 2;
      ray.ray.at(mid, p);
      if (p.y <= surface(p.x, p.z)) hi = mid; else lo = mid;
    }
    ray.ray.at(hi, p);
    if (Math.abs(p.x) > W / 2 || Math.abs(p.z) > H / 2) return null;
    return { u: (p.x + W / 2) / W, v: (p.z + H / 2) / H };
  };
  const cursor = $("cursor");
  const ndc = new THREE.Vector2();
  let pending: PointerEvent | null = null;
  renderer.domElement.addEventListener("pointermove", (e) => (pending = e));
  renderer.domElement.addEventListener("pointerleave", () => { pending = null; cursor.hidden = true; });
  const masksReady = new Map<string, { n: number; r: Uint8Array; g: Uint8Array }>();
  const readout = () => {
    const e = pending;
    pending = null;
    if (!e) return;
    if (e.buttons) { cursor.hidden = true; return; }
    ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = pick();
    if (!hit) { cursor.hidden = true; return; }
    const { u, v } = hit;
    const gi = Math.min(G - 1, Math.round(v * (G - 1))) * G + Math.min(G - 1, Math.round(u * (G - 1)));
    const at = (url: string) => {
      const m = masksReady.get(url);
      if (!m) { maskOf(url).then((x) => masksReady.set(url, x)); return null; }
      const k = Math.min(m.n - 1, Math.floor(v * m.n)) * m.n + Math.min(m.n - 1, Math.floor(u * m.n));
      return { water: m.r[k] > 127, nodata: m.g[k] > 127 };
    };
    let status = "sin análisis IA";
    const a = analysisAt(st.day);
    if (a) {
      const now = at(waterUrl(a));
      const pre = meta.pre_event[st.year];
      const before = pre ? at(`${BASE}water_${pre}.png`) : null;
      status = !now ? "…" : now.nodata ? "sin dato (nube o fuera de pasada)" : now.water ? (before?.water ? "agua ya presente" : pre ? "<b>inundado</b>" : "agua") : "seco";
      status += ` · ${a.label}`;
    }
    const [lon0, lat0, lon1, lat1] = meta.bounds_wgs84;
    const lon = lon0 + u * (lon1 - lon0), lat = lat1 - v * (lat1 - lat0);
    cursor.innerHTML = `${heights[gi].toFixed(0)} m s. n. m.<br>${status}<br>${Math.abs(lat).toFixed(3)}° S ${Math.abs(lon).toFixed(3)}° O`;
    cursor.style.left = `${e.clientX}px`;
    cursor.style.top = `${e.clientY}px`;
    cursor.hidden = false;
  };

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    drawTrack();
  });

  // ---------------------------------------------------------------- start
  await setYear("2023");

  let miniKey = "", frames = 0, slow = 0, last = performance.now();
  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const now = performance.now(), dt = now - last;
    if (dt > 22) slow++;
    frames++;
    last = now;
    if (frames === 90) {
      if (slow > 30 && ratio > 1) { ratio = Math.max(1, ratio - 0.5); renderer.setPixelRatio(ratio); }
      frames = slow = 0;
    }
    if (st.playing) {
      acc += (Math.min(dt, 100) / 1000) * st.speed;
      if (acc >= 1) {
        const k = Math.floor(acc);
        acc -= k;
        if (st.day + k >= range()[1]) { setDay(range()[1]); setPlaying(false); } else setDay(st.day + k);
      }
    }
    for (let i = fades.length - 1; i >= 0; i--) {
      const f = fades[i], t = Math.min(1, (now - f.t0) / f.ms);
      f.u.value = t * t * (3 - 2 * t);
      if (t >= 1) fades.splice(i, 1);
    }
    if (camMove) {
      const t = Math.min(1, (now - camMove.t0) / 800), e = t * t * (3 - 2 * t);
      camera.position.lerpVectors(camMove.from, camMove.to, e);
      if (t >= 1) camMove = null;
    }
    waterMat.uniforms.uTime.value = clock.getElapsedTime();
    controls.update();
    const key = [camera.position.x, camera.position.y, camera.position.z, controls.target.x, controls.target.z].map((n) => n.toFixed(3)).join(",") + st.ground;
    if (key !== miniKey || miniDirty) { miniKey = key; miniDirty = false; drawMini(); }
    readout();
    renderer.render(scene, camera);
  });
}

main().catch((e) => console.error("terrain:", e));
