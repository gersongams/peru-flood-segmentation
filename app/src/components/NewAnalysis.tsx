import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { api, type Acquisition, type Job, type StageId } from "../api";
import { EVENTS, TILES, type EventId } from "../config";
import { fmtDate, mmss } from "../format";

interface Props {
  eventId: EventId;
  onEvent: (e: EventId) => void;
  onDone: (scene: string) => void;
}

const STAGES: { id: StageId; name: string; detail: string }[] = [
  { id: "auth", name: "Autenticación en Copernicus", detail: "Copernicus Data Space Ecosystem" },
  { id: "catalog", name: "Búsqueda en el catálogo", detail: "Producto S2 L1C de la tesela y la fecha" },
  { id: "download", name: "Descarga del producto", detail: "Archivo SAFE completo" },
  { id: "read", name: "Lectura de bandas", detail: "13 bandas a 10 m, corrección de offset" },
  { id: "mask", name: "Máscara de nubes y sin datos", detail: "Regla de nube espesa del proyecto" },
  { id: "inference", name: "Inferencia del modelo", detail: "U-Net entrenada con Perú · 3 semillas" },
  { id: "write", name: "Escritura de GeoTIFF", detail: "Probabilidad, incertidumbre, máscara" },
];
const TARGET_S = 15 * 60;

type Bounds = [number, number, number, number];

export default function NewAnalysis({ eventId, onEvent, onDone }: Props) {
  const ev = EVENTS[eventId];
  const [tile, setTile] = useState(TILES[eventId][0].id);
  const [acqs, setAcqs] = useState<Acquisition[] | null>(null);
  const [acqError, setAcqError] = useState<string | null>(null);
  const [date, setDate] = useState<string | null>(null);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => setTile(TILES[eventId][0].id), [eventId]);

  useEffect(() => {
    let alive = true;
    setAcqs(null);
    setAcqError(null);
    api.acquisitions(tile, ev.range[0], ev.range[1])
      .then((a) => {
        if (!alive) return;
        // One row per date (a reprocessed product can appear twice).
        const seen = new Set<string>();
        const uniq = a.filter((x) => (seen.has(x.date) ? false : (seen.add(x.date), true)));
        setAcqs(uniq);
        setDate((d) => (d && uniq.some((x) => x.date === d) ? d : uniq.find((x) => ev.dates.includes(x.date))?.date ?? uniq[0]?.date ?? null));
      })
      .catch((e) => alive && setAcqError(String(e)));
    return () => {
      alive = false;
    };
  }, [tile, ev]);

  // Poll the running job.
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const t = window.setInterval(async () => setJob(await api.job(job.id)), 1000);
    return () => window.clearInterval(t);
  }, [job]);

  const submit = async () => {
    if (!date) return;
    setSubmitError(null);
    try {
      setJob(await api.createJob({ event: eventId, tile, date, bounds: bounds ?? undefined }));
    } catch (e) {
      setSubmitError(String(e));
    }
  };

  const running = job && (job.status === "queued" || job.status === "running");
  const elapsed = job ? (job.status === "done" || job.status === "error" ? job.seconds ?? 0 : job.elapsed ?? 0) : 0;
  const doneFrac = job ? STAGES.reduce((a, s) => a + (job.stages[s.id].status === "pending" ? 0 : job.stages[s.id].fraction), 0) / STAGES.length : 0;

  return (
    <div className="page">
      <form className="form" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <div>
          <h1>Nuevo análisis</h1>
          <p>Elige el evento, la tesela y la fecha. El sistema descarga la imagen de Copernicus y genera el mapa de agua de la tesela completa.</p>
        </div>

        <fieldset style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <legend><span className="step">1</span>Evento</legend>
          {(Object.keys(EVENTS) as EventId[]).map((id) => (
            <label key={id} className={`choice ${id === eventId ? "on" : ""}`}>
              <input type="radio" name="ev" checked={id === eventId} onChange={() => onEvent(id)} />
              <span>
                <b>{EVENTS[id].name}</b>
                <small>{EVENTS[id].location} · referencia {id === "yaku2023" ? "UNOSAT" : "Copernicus EMS"}</small>
              </span>
            </label>
          ))}
        </fieldset>

        <fieldset>
          <legend><span className="step">2</span>Tesela y área</legend>
          <p className="hint" style={{ margin: "0 0 8px" }}>Haz clic en una tesela del mapa o arrastra con Mayús para dibujar un rectángulo. Se procesa la tesela entera.</p>
          <div className="codebox">
            Tesela {tile}{TILES[eventId].find((t) => t.id === tile)?.share ? ` · ${TILES[eventId].find((t) => t.id === tile)!.share} del área UNOSAT` : ""}
            <br />
            {bounds ? `Vista: ${bounds.map((v) => v.toFixed(3)).join(", ")}` : "Sin rectángulo: vista de la tesela completa"}
          </div>
        </fieldset>

        <fieldset>
          <legend><span className="step">3</span>Fecha de la imagen Sentinel-2</legend>
          {acqError && <div className="error">No se pudo consultar el catálogo: {acqError}</div>}
          {!acqs && !acqError && <div className="hint">Consultando el catálogo de Copernicus…</div>}
          {acqs && (
            <div className="list">
              {acqs.map((a) => (
                <label key={a.Id} className={a.date === date ? "on" : ""}>
                  <input type="radio" name="fecha" checked={a.date === date} onChange={() => setDate(a.date)} />
                  <span>
                    {fmtDate(a.date)}
                    <small>
                      {a.cloud_cover != null ? `nubosidad de la escena ${Math.round(a.cloud_cover)} %` : ""}
                      {ev.dates.includes(a.date) ? " · con mapa de referencia" : ""}
                    </small>
                  </span>
                  <span className={`pill ${a.cached ? "ok" : ""}`}>{a.cached ? "En caché" : "Requiere descarga"}</span>
                </label>
              ))}
            </div>
          )}
        </fieldset>

        <div className="note">
          <b>Imagen pre-evento.</b> El área inundada se calcula contra la imagen más despejada de la misma tesela en los 60 días anteriores al inicio del evento ({fmtDate(EVENTS[eventId].events[0]?.date ?? ev.range[0])}).
        </div>

        {submitError && <div className="error">{submitError}</div>}
        <button type="submit" className="btn primary" disabled={!date || !!running}>
          {running ? "Procesando…" : `Procesar tesela ${tile}`}
        </button>
      </form>

      <section className="picker">
        <TilePicker eventId={eventId} tile={tile} onTile={setTile} bounds={bounds} onBounds={setBounds} />
      </section>

      <aside className="job">
        {!job && (
          <div className="note">
            Aquí verás el avance del procesamiento: descarga, lectura, inferencia y escritura. Una tesela ya procesada se abre desde la caché sin descargarla.
          </div>
        )}
        {job && (
          <>
            <div>
              <div className="kicker">{job.status === "done" ? "Procesamiento terminado" : job.status === "error" ? "Error" : "Procesamiento en curso"}</div>
              <div style={{ fontWeight: 700, fontSize: 16, marginTop: 2 }}>{job.tile} · {fmtDate(job.date)}</div>
            </div>
            <div className="timer">
              <div>
                <small>{job.cached ? "Cargado desde caché" : "Tiempo transcurrido"}</small>
                <div className="big">{mmss(elapsed)}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <small>Objetivo</small>
                <span className="mono" style={{ fontSize: 16 }}>≤ {mmss(TARGET_S)}</span>
              </div>
            </div>
            {job.cached && <div className="note">Esta tesela y fecha ya estaban procesadas: se abren desde la caché, sin descargar ni volver a ejecutar el modelo.</div>}
            {!job.cached && <div className="progress"><div style={{ width: `${Math.round(doneFrac * 100)}%` }} /></div>}
            {!job.cached && <ol className="stages">
              {STAGES.map((s) => {
                const st = job.stages[s.id];
                return (
                  <li key={s.id} className={st.status}>
                    <span className="ic">{st.status === "done" || st.status === "cached" ? "✓" : ""}</span>
                    <span className="nm">
                      {s.name}
                      <small>{st.status === "running" && st.detail ? st.detail : s.detail}</small>
                      {st.status === "running" && <div className="bar"><div style={{ width: `${Math.round(st.fraction * 100)}%` }} /></div>}
                    </span>
                    <span className="tm">{st.seconds != null ? mmss(st.seconds) : ""}</span>
                  </li>
                );
              })}
            </ol>}
            {job.status === "error" && <div className="error">{job.error}</div>}
            {job.status === "done" && job.scene && (
              <button className="btn dark" onClick={() => onDone(job.scene!)}>Ver en el visor</button>
            )}
            {running && <div className="note">Puedes cambiar de pestaña: el trabajo sigue en el servidor.</div>}
          </>
        )}
      </aside>
    </div>
  );
}

function TilePicker({ eventId, tile, onTile, bounds, onBounds }: { eventId: EventId; tile: string; onTile: (t: string) => void; bounds: Bounds | null; onBounds: (b: Bounds | null) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const onTileRef = useRef(onTile);
  const onBoundsRef = useRef(onBounds);
  onTileRef.current = onTile;
  onBoundsRef.current = onBounds;

  useEffect(() => {
    const ev = EVENTS[eventId];
    const m = new maplibregl.Map({
      container: ref.current!,
      style: {
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        sources: { base: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 16, attribution: "Esri, HERE, Garmin, © OpenStreetMap" } },
        layers: [{ id: "base", type: "raster", source: "base", paint: { "raster-saturation": -0.4 } }],
      },
      center: ev.center,
      zoom: 7.2,
      boxZoom: false,
    });
    m.on("load", () => setReady(true));
    // Shift + drag draws the view rectangle.
    let start: maplibregl.LngLat | null = null;
    m.on("mousedown", (e) => {
      if (!e.originalEvent.shiftKey) return;
      e.preventDefault();
      m.dragPan.disable();
      start = e.lngLat;
    });
    m.on("mousemove", (e) => {
      if (!start) return;
      const b: Bounds = [Math.min(start.lng, e.lngLat.lng), Math.min(start.lat, e.lngLat.lat), Math.max(start.lng, e.lngLat.lng), Math.max(start.lat, e.lngLat.lat)];
      (m.getSource("rect") as maplibregl.GeoJSONSource | undefined)?.setData(rectGeo(b));
    });
    m.on("mouseup", (e) => {
      if (!start) return;
      const b: Bounds = [Math.min(start.lng, e.lngLat.lng), Math.min(start.lat, e.lngLat.lat), Math.max(start.lng, e.lngLat.lng), Math.max(start.lat, e.lngLat.lat)];
      start = null;
      m.dragPan.enable();
      onBoundsRef.current(b);
      const c = m.project([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]);
      const f = m.queryRenderedFeatures(c, { layers: ["tiles-fill"] })[0];
      if (f) onTileRef.current(f.properties.tile);
    });
    m.on("click", "tiles-fill", (e) => {
      if (e.originalEvent.shiftKey) return;
      const t = e.features?.[0]?.properties?.tile;
      if (t) {
        onTileRef.current(t);
        onBoundsRef.current(null);
      }
    });
    m.on("mouseenter", "tiles-fill", () => (m.getCanvas().style.cursor = "pointer"));
    m.on("mouseleave", "tiles-fill", () => (m.getCanvas().style.cursor = ""));
    map.current = m;
    return () => m.remove();
  }, [eventId]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const ids = TILES[eventId].map((t) => t.id).join(",");
    fetch(`/api/footprints?tiles=${ids}`).then((r) => r.json()).then((fc) => {
      for (const f of fc.features) f.properties.share = TILES[eventId].find((t) => t.id === f.properties.tile)?.share ?? "";
      const xs = fc.features.flatMap((f: GeoJSON.Feature<GeoJSON.Polygon>) => f.geometry.coordinates[0]);
      m.fitBounds([Math.min(...xs.map((c: number[]) => c[0])), Math.min(...xs.map((c: number[]) => c[1])), Math.max(...xs.map((c: number[]) => c[0])), Math.max(...xs.map((c: number[]) => c[1]))], { padding: 40, duration: 0 });
      if (m.getSource("tiles")) (m.getSource("tiles") as maplibregl.GeoJSONSource).setData(fc);
      else {
        m.addSource("tiles", { type: "geojson", data: fc });
        m.addLayer({ id: "tiles-fill", type: "fill", source: "tiles", paint: { "fill-color": "#2563b8", "fill-opacity": ["case", ["==", ["get", "tile"], tile], 0.22, 0.06] } });
        m.addLayer({ id: "tiles-line", type: "line", source: "tiles", paint: { "line-color": ["case", ["==", ["get", "tile"], tile], "#1d4f9a", "#5f564c"], "line-width": ["case", ["==", ["get", "tile"], tile], 3, 1.2] } });
        m.addLayer({ id: "tiles-label", type: "symbol", source: "tiles", layout: { "text-field": ["format", ["get", "tile"], { "font-scale": 1 }, "\n", {}, ["get", "share"], { "font-scale": 0.8 }], "text-font": ["Noto Sans Regular"], "text-size": 13 }, paint: { "text-color": "#2a2520", "text-halo-color": "#fff", "text-halo-width": 1.5 } });
        m.addSource("rect", { type: "geojson", data: rectGeo(null) });
        m.addLayer({ id: "rect-fill", type: "fill", source: "rect", paint: { "fill-color": "#ffffff", "fill-opacity": 0.2 } });
        m.addLayer({ id: "rect-line", type: "line", source: "rect", paint: { "line-color": "#c8553d", "line-width": 2.5, "line-dasharray": [3, 2] } });
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, eventId]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !m.getLayer("tiles-fill")) return;
    m.setPaintProperty("tiles-fill", "fill-opacity", ["case", ["==", ["get", "tile"], tile], 0.22, 0.06]);
    m.setPaintProperty("tiles-line", "line-color", ["case", ["==", ["get", "tile"], tile], "#1d4f9a", "#5f564c"]);
    m.setPaintProperty("tiles-line", "line-width", ["case", ["==", ["get", "tile"], tile], 3, 1.2]);
  }, [tile, ready]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    (m.getSource("rect") as maplibregl.GeoJSONSource | undefined)?.setData(rectGeo(bounds));
  }, [bounds, ready]);

  return (
    <>
      <div ref={ref} className="map" />
      <div className="ov floating" style={{ top: 14, left: 14, padding: "10px 12px", maxWidth: 280, fontSize: 12 }}>
        <b>Teselas Sentinel-2 (MGRS, ~110 km)</b>
        <div className="muted">El rectángulo define la vista; el sistema procesa la tesela que lo contiene.</div>
      </div>
    </>
  );
}

function rectGeo(b: Bounds | null): GeoJSON.FeatureCollection {
  if (!b) return { type: "FeatureCollection", features: [] };
  const [w, s, e, n] = b;
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }] };
}
