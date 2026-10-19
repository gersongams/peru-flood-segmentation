import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { api, type Pixel, type Scene } from "../api";
import { LAYERS } from "../config";

export interface LayerState {
  on: boolean;
  opacity: number;
}

interface Props {
  scene: Scene | null;
  layers: Record<string, LayerState>;
  layerIds?: string[]; // drawn bottom to top; default: the viewer layers
  center: [number, number];
  zoom: number;
  onView?: (bounds: [number, number, number, number]) => void;
}

const ALL_IDS = ["rgb", "probability", "compare", "uncertainty", "mask", "reference", "errors", "flood"];

const BASEMAP: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    base: {
      type: "raster",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "Esri, HERE, Garmin, © OpenStreetMap · Contiene datos modificados de Copernicus Sentinel",
      maxzoom: 16,
    },
    labels: {
      type: "raster",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"],
      maxzoom: 16,
      tileSize: 256,
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#e3d8c4" } },
    { id: "base", type: "raster", source: "base", paint: { "raster-saturation": -0.4, "raster-opacity": 0.85 } },
    { id: "labels-anchor", type: "background", paint: { "background-opacity": 0 } },
    { id: "labels", type: "raster", source: "labels" },
  ],
};

export default function MapView({ scene, layers, layerIds = LAYERS.map((l) => l.id), center, zoom, onView }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [hover, setHover] = useState<{ x: number; y: number; lng: number; lat: number; px: Pixel } | null>(null);
  const sceneRef = useRef<Scene | null>(scene);
  sceneRef.current = scene;

  useEffect(() => {
    const m = new maplibregl.Map({ container: ref.current!, style: BASEMAP, center, zoom, attributionControl: { compact: true } });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-left");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    m.on("load", () => setReady(true));
    const emit = () => {
      const b = m.getBounds();
      onView?.([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
    };
    m.on("moveend", emit);
    let timer = 0;
    let seq = 0;
    m.on("mousemove", (e) => {
      window.clearTimeout(timer);
      const s = sceneRef.current;
      if (!s) return;
      const { lng, lat } = e.lngLat;
      const { x, y } = e.point;
      timer = window.setTimeout(async () => {
        const mine = ++seq;
        try {
          const px = await api.pixel(s.scene, lng, lat);
          if (mine === seq) setHover(px.inside ? { x, y, lng, lat, px } : null);
        } catch {
          setHover(null);
        }
      }, 120);
    });
    m.on("mouseout", () => {
      window.clearTimeout(timer);
      setHover(null);
    });
    map.current = m;
    return () => m.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // One raster source per layer of the current scene.
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    for (const id of ALL_IDS) {
      if (m.getLayer(id)) m.removeLayer(id);
      if (m.getSource(id)) m.removeSource(id);
    }
    if (!scene) return;
    for (const id of layerIds) {
      m.addSource(id, { type: "raster", tiles: [api.tileUrl(scene.scene, id)], tileSize: 256, bounds: scene.bounds, minzoom: 6, maxzoom: 15 });
      m.addLayer({ id, type: "raster", source: id, paint: { "raster-fade-duration": 0, "raster-resampling": id === "rgb" ? "linear" : "nearest" } }, "labels-anchor");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, ready, layerIds.join()]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !scene) return;
    for (const id of layerIds) {
      if (!m.getLayer(id) || !layers[id]) continue;
      m.setLayoutProperty(id, "visibility", layers[id].on ? "visible" : "none");
      m.setPaintProperty(id, "raster-opacity", layers[id].opacity);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, scene, ready, layerIds.join()]);

  useEffect(() => {
    if (map.current && scene) map.current.fitBounds(scene.bounds, { padding: 30, duration: 600 });
  }, [scene]);

  return (
    <>
      <div ref={ref} className="map" />
      {hover && (
        <div className="ov tooltip" style={{ left: hover.x, top: hover.y }}>
          <Readout px={hover.px} />
          <div className="c">
            {Math.abs(hover.lat).toFixed(4)}° {hover.lat < 0 ? "S" : "N"} · {Math.abs(hover.lng).toFixed(4)}° {hover.lng < 0 ? "O" : "E"}
          </div>
        </div>
      )}
    </>
  );
}

function Readout({ px }: { px: Pixel }) {
  if (px.state === "nodata") return <div className="t">Sin datos</div>;
  const p = px.probability ?? 0;
  const water = p > 0.5;
  return (
    <>
      <div className="t">
        <span className="dot" style={{ background: water ? "#5da6e6" : "#c9b48a" }} />
        {water ? "Agua detectada" : "Sin agua"}
        {px.state === "cloud" && <span style={{ color: "#e9a21b", fontWeight: 500 }}> · nube</span>}
      </div>
      <div className="g">
        <span>Probabilidad</span>
        <span>{p.toFixed(2)}</span>
        <span>Incertidumbre (σ)</span>
        <span>{(px.uncertainty ?? 0).toFixed(3)}</span>
      </div>
    </>
  );
}
