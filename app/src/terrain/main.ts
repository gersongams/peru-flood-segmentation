// 3D spike: Copernicus DEM relief of Bajo Piura / Sechura with the AI water per Sentinel-2 date.
// Data from scripts/system/terrain_export.py (app/public/terrain/<zone>/).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./terrain.css";

interface Meta {
  title: string;
  size_m: [number, number];
  grid: number;
  bounds_wgs84: [number, number, number, number];
  height_min: number;
  height_max: number;
  dates: { date: string; label: string; water_km2: number }[];
}

const BASE = `${import.meta.env.BASE_URL}terrain/piura/`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main() {
  const meta: Meta = await (await fetch(`${BASE}meta.json`)).json();
  const heights = new Float32Array(await (await fetch(`${BASE}height.bin`)).arrayBuffer());
  const G = meta.grid;
  const W = meta.size_m[0] / 1000; // km
  const H = meta.size_m[1] / 1000;
  $("t-title").textContent = meta.title;

  // ---------- renderer, scene, camera
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  $("scene").appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#e9e2d6");
  scene.fog = new THREE.Fog("#e9e2d6", 70, 140);

  const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 400);
  camera.position.set(4, 44, 62);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 2);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.minDistance = 6;
  controls.maxDistance = 110;

  scene.add(new THREE.HemisphereLight("#fff6e8", "#8a7a63", 1.1));
  const sun = new THREE.DirectionalLight("#fff1dc", 2.2);
  sun.position.set(-40, 22, -25); // low sun from the north-west: dunes and terraces read well
  scene.add(sun);

  // ---------- terrain geometry (x east, z south; row 0 of the grid is north)
  const geo = new THREE.PlaneGeometry(W, H, G - 1, G - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  let exag = Number(($("t-exag") as HTMLInputElement).value);
  const applyHeights = () => {
    for (let i = 0; i < G * G; i++) pos.setY(i, (heights[i] / 1000) * exag);
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    $("t-exag-v").textContent = `×${exag}`;
  };
  applyHeights();

  // Relief colours: sand ramp by elevation.
  const colors = new Float32Array(G * G * 3);
  const ramp = [new THREE.Color("#e2d3b3"), new THREE.Color("#d2b88d"), new THREE.Color("#b8966a"), new THREE.Color("#8c6e4f")];
  const span = Math.max(1, meta.height_max - meta.height_min);
  for (let i = 0; i < G * G; i++) {
    const t = Math.min(0.999, Math.max(0, (heights[i] - meta.height_min) / span)) * (ramp.length - 1);
    const c = ramp[Math.floor(t)].clone().lerp(ramp[Math.floor(t) + 1], t % 1);
    colors.set([c.r, c.g, c.b], i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const loader = new THREE.TextureLoader();
  const load = (url: string) =>
    new Promise<THREE.Texture>((res) => loader.load(url, (t) => { t.anisotropy = renderer.capabilities.getMaxAnisotropy(); res(t); }));
  const rgb = await load(`${BASE}rgb.jpg`);
  rgb.colorSpace = THREE.SRGBColorSpace;

  const satMat = new THREE.MeshStandardMaterial({ map: rgb, roughness: 1, metalness: 0 });
  const reliefMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 });
  const ground = new THREE.Mesh(geo, satMat);
  scene.add(ground);

  // Skirt: a dark base under the block so it reads as a cut-out model.
  const base = new THREE.Mesh(new THREE.BoxGeometry(W, 0.6, H), new THREE.MeshStandardMaterial({ color: "#5b4d3d", roughness: 1 }));
  base.position.y = -0.3;
  scene.add(base);

  // ---------- water: same surface, cross-fade between two dates, new vs pre-existing water
  const waterTex = await Promise.all(meta.dates.map((d) => load(`${BASE}water_${d.date}.png`)));
  const pre = waterTex[0];
  const waterMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    uniforms: {
      uA: { value: waterTex[0] }, uB: { value: waterTex[0] }, uPre: { value: pre }, uT: { value: 0 },
      uTime: { value: 0 }, uSun: { value: sun.position.clone().normalize() },
      uNew: { value: new THREE.Color("#1d4f9a") }, uOld: { value: new THREE.Color("#8fbbe3") },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() {
        vUv = uv; vN = normalize(normalMatrix * normal);
        vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D uA, uB, uPre; uniform float uT, uTime; uniform vec3 uSun, uNew, uOld;
      varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() {
        float a = mix(texture2D(uA, vUv).r, texture2D(uB, vUv).r, uT);
        float alpha = smoothstep(0.35, 0.65, a);
        if (alpha < 0.01) discard;
        float before = smoothstep(0.35, 0.65, texture2D(uPre, vUv).r);
        vec3 col = mix(uNew, uOld, before);
        float ripple = 0.04 * sin(vW.x * 9.0 + uTime * 1.3) * sin(vW.z * 7.0 - uTime * 1.1);
        vec3 v = normalize(cameraPosition - vW);
        float spec = pow(max(dot(reflect(-uSun, vec3(0.0, 1.0, 0.0)), v), 0.0), 40.0);
        col = col * (0.92 + ripple) + vec3(0.9, 0.95, 1.0) * spec * 0.35;
        gl_FragColor = vec4(col, 0.9 * alpha);
        #include <colorspace_fragment>
      }`,
  });
  const water = new THREE.Mesh(geo, waterMat);
  water.renderOrder = 1;
  scene.add(water);

  // Pixel data of the water masks for the cursor readout.
  const masks = await Promise.all(meta.dates.map(async (d) => {
    const img = await createImageBitmap(await (await fetch(`${BASE}water_${d.date}.png`)).blob());
    const c = new OffscreenCanvas(img.width, img.height);
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    return { w: img.width, h: img.height, data: ctx.getImageData(0, 0, img.width, img.height).data };
  }));

  // ---------- minimap: satellite image + water of the current date + camera footprint
  const mini = $("t-mini") as HTMLCanvasElement;
  const mctx = mini.getContext("2d")!;
  const M = mini.width;
  const rgbImg = await createImageBitmap(await (await fetch(`${BASE}rgb.jpg`)).blob(), { resizeWidth: M, resizeHeight: M });
  // Water layers at minimap resolution (downscale the masks first; ~190 k pixels each).
  const small = await Promise.all(meta.dates.map(async (d) => {
    const bmp = await createImageBitmap(await (await fetch(`${BASE}water_${d.date}.png`)).blob(), { resizeWidth: M, resizeHeight: M, resizeQuality: "medium" });
    const c = new OffscreenCanvas(M, M);
    const ctx = c.getContext("2d")!;
    ctx.drawImage(bmp, 0, 0);
    return ctx.getImageData(0, 0, M, M).data;
  }));
  const waterLayers = small.map((m, k) => {
    const c = new OffscreenCanvas(M, M);
    const img = new ImageData(M, M);
    for (let i = 0; i < M * M; i++) {
      if (m[i * 4] <= 127) continue;
      const old = k > 0 && small[0][i * 4] > 127;
      const o = i * 4;
      img.data[o] = old ? 143 : 29; img.data[o + 1] = old ? 187 : 79; img.data[o + 2] = old ? 227 : 154; img.data[o + 3] = 235;
    }
    c.getContext("2d")!.putImageData(img, 0, 0);
    return c;
  });
  // World (x east, z south, km, centred) <-> minimap pixels.
  const toMini = (x: number, z: number): [number, number] => [((x + W / 2) / W) * M, ((z + H / 2) / H) * M];
  const ground0 = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const corner = new THREE.Vector3();
  const footprint = (): [number, number][] => {
    const pts: [number, number][] = [];
    for (const [cx, cy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      ray.setFromCamera(new THREE.Vector2(cx, cy), camera);
      if (ray.ray.intersectPlane(ground0, corner) && corner.distanceTo(camera.position) < 300) pts.push(toMini(corner.x, corner.z));
      else {
        // Ray above the horizon: clamp to a far point along its horizontal direction.
        const d = ray.ray.direction.clone().setY(0).normalize().multiplyScalar(200).add(camera.position);
        pts.push(toMini(d.x, d.z));
      }
    }
    return pts;
  };
  const drawMini = () => {
    mctx.clearRect(0, 0, M, M);
    mctx.drawImage(rgbImg, 0, 0);
    mctx.globalAlpha = 0.9;
    mctx.drawImage(waterLayers[current], 0, 0);
    mctx.globalAlpha = 1;
    const fp = footprint();
    mctx.save();
    mctx.beginPath();
    mctx.rect(0, 0, M, M);
    mctx.clip();
    mctx.beginPath();
    fp.forEach(([x, y], k) => (k ? mctx.lineTo(x, y) : mctx.moveTo(x, y)));
    mctx.closePath();
    mctx.fillStyle = "rgba(255, 255, 255, 0.16)";
    mctx.fill();
    mctx.lineWidth = 5;
    mctx.strokeStyle = "#c8553d";
    mctx.stroke();
    mctx.restore();
    // Camera position and heading.
    const [px, py] = toMini(camera.position.x, camera.position.z);
    const [tx, ty] = toMini(controls.target.x, controls.target.z);
    const ang = Math.atan2(ty - py, tx - px);
    const cx = Math.min(M - 10, Math.max(10, px)), cy = Math.min(M - 10, Math.max(10, py));
    mctx.save();
    mctx.translate(cx, cy);
    mctx.rotate(ang);
    mctx.beginPath();
    mctx.moveTo(14, 0); mctx.lineTo(-8, -9); mctx.lineTo(-4, 0); mctx.lineTo(-8, 9); mctx.closePath();
    mctx.fillStyle = "#2a2520";
    mctx.strokeStyle = "#fff";
    mctx.lineWidth = 2;
    mctx.fill(); mctx.stroke();
    mctx.restore();
    mctx.beginPath();
    mctx.arc(tx, ty, 5, 0, Math.PI * 2);
    mctx.fillStyle = "#c8553d";
    mctx.fill();
  };
  mini.addEventListener("click", (e) => {
    const r = mini.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W - W / 2;
    const z = ((e.clientY - r.top) / r.height) * H - H / 2;
    const dx = x - controls.target.x, dz = z - controls.target.z;
    controls.target.x += dx; controls.target.z += dz;
    camera.position.x += dx; camera.position.z += dz;
  });

  // ---------- date controls and playback
  let current = 1 < meta.dates.length ? 1 : 0;
  const datesEl = $("t-dates");
  const buttons = meta.dates.map((d, i) => {
    const b = document.createElement("button");
    b.textContent = d.label;
    b.onclick = () => go(i);
    datesEl.appendChild(b);
    return b;
  });
  let anim: { from: number; to: number; t0: number } | null = null;
  const go = (i: number) => {
    waterMat.uniforms.uA.value = waterTex[current];
    waterMat.uniforms.uB.value = waterTex[i];
    waterMat.uniforms.uT.value = 0;
    anim = { from: current, to: i, t0: performance.now() };
    current = i;
    $("t-mini-date").textContent = meta.dates[i].label;
    buttons.forEach((b, k) => b.classList.toggle("on", k === i));
    const d = meta.dates[i];
    $("t-water").textContent = `${d.water_km2.toLocaleString("es-PE")} km²`;
    $("t-new").textContent = i === 0 ? "—" : `${Math.max(0, d.water_km2 - meta.dates[0].water_km2).toLocaleString("es-PE", { maximumFractionDigits: 1 })} km²`;
  };
  waterMat.uniforms.uA.value = waterTex[current];
  waterMat.uniforms.uB.value = waterTex[current];
  go(current);

  let playing = 0;
  $("t-play").onclick = () => {
    if (playing) { window.clearInterval(playing); playing = 0; $("t-play").textContent = "▶ Reproducir"; return; }
    go(0);
    let k = 0;
    playing = window.setInterval(() => {
      k = (k + 1) % meta.dates.length;
      go(k);
    }, 2200);
    $("t-play").textContent = "❚❚ Pausa";
  };

  ($("t-exag") as HTMLInputElement).oninput = (e) => {
    exag = Number((e.target as HTMLInputElement).value);
    applyHeights();
  };
  $("t-style").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      ground.material = b.dataset.style === "relief" ? reliefMat : satMat;
      $("t-style").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    };
  });

  // ---------- cursor readout: elevation, water at this date and before the event
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const tip = $("t-cursor");
  renderer.domElement.addEventListener("pointermove", (e) => {
    ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(ground)[0];
    if (!hit || !hit.uv) { tip.hidden = true; return; }
    const u = hit.uv.x, v = 1 - hit.uv.y; // v = 0 at the north edge
    const gi = Math.min(G - 1, Math.round(v * (G - 1))) * G + Math.min(G - 1, Math.round(u * (G - 1)));
    const at = (m: (typeof masks)[number]) => m.data[(Math.min(m.h - 1, Math.floor(v * m.h)) * m.w + Math.min(m.w - 1, Math.floor(u * m.w))) * 4] > 127;
    const [lon0, lat0, lon1, lat1] = meta.bounds_wgs84;
    const lon = lon0 + u * (lon1 - lon0), lat = lat1 - v * (lat1 - lat0);
    const now = at(masks[current]), before = at(masks[0]);
    tip.innerHTML = `${heights[gi].toFixed(0)} m s. n. m.<br>${now ? (before ? "agua ya presente" : "<b>inundado</b>") : "seco"} · ${meta.dates[current].label}<br>${Math.abs(lat).toFixed(3)}° S ${Math.abs(lon).toFixed(3)}° O`;
    tip.style.left = `${e.clientX}px`;
    tip.style.top = `${e.clientY}px`;
    tip.hidden = false;
  });
  renderer.domElement.addEventListener("pointerleave", () => (tip.hidden = true));

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    waterMat.uniforms.uTime.value = clock.getElapsedTime();
    if (anim) {
      const t = Math.min(1, (performance.now() - anim.t0) / 1200);
      waterMat.uniforms.uT.value = t * t * (3 - 2 * t);
      if (t >= 1) anim = null;
    }
    controls.update();
    drawMini();
    renderer.render(scene, camera);
  });
}

main().catch((e) => console.error("terrain:", e));
