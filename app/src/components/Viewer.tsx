import { useEffect, useState } from "react";
import { api, type Flood, type Scene } from "../api";
import { LAYERS, type FloodEvent, type LayerId } from "../config";
import { fmtDate, km2, num } from "../format";
import { Info } from "./Icons";
import MapView, { type LayerState } from "./MapView";
import Timeline from "./Timeline";

interface Props {
  event: FloodEvent;
  scenes: Scene[]; // processed scenes of this event, oldest first
  scene: Scene | null;
  onScene: (s: Scene) => void;
  onNew: () => void;
  onView?: (b: [number, number, number, number]) => void;
}

const initial = Object.fromEntries(LAYERS.map((l) => [l.id, { on: l.defaultOn, opacity: l.defaultOpacity }])) as Record<LayerId, LayerState>;

export default function Viewer({ event, scenes, scene, onScene, onNew, onView }: Props) {
  const [layers, setLayers] = useState(initial);
  const set = (id: LayerId, patch: Partial<LayerState>) => setLayers((s) => ({ ...s, [id]: { ...s[id], ...patch } }));
  const [flood, setFlood] = useState<Flood | null>(null);
  useEffect(() => {
    setFlood(null);
    if (scene) api.flood(scene.scene).then(setFlood).catch(() => setFlood(null));
  }, [scene]);
  const px = scene?.pixels;
  const total = px ? px.valid + px.cloud + px.nodata : 0;

  return (
    <div className="viewer">
      <div className="main">
        <div className="mapwrap">
          <MapView scene={scene} layers={layers} center={event.center} zoom={event.zoom} onView={onView} />
          {scene ? (
            <div className="ov floating datechip" style={{ left: 56 }}>
              <div className="d">{fmtDate(scene.date)}</div>
              <div className="s">Sentinel-2 L1C · tesela {scene.tile}</div>
            </div>
          ) : (
            <div className="empty-map">
              <div className="floating">
                <b>Aún no hay escenas procesadas para este evento</b>
                <p className="muted">Elige una fecha y una tesela en «Nuevo análisis». El sistema descarga la imagen y genera el mapa.</p>
                <button className="btn dark" onClick={onNew}>Nuevo análisis</button>
              </div>
            </div>
          )}
          <div className="ov disclaimer">
            <Info />
            Prototipo de investigación. Referencia preliminar, sin validación de campo.
          </div>
          {scene && (
            <div className="ov floating legend">
              <b>Probabilidad de agua</b>
              <div className="bar" style={{ background: "linear-gradient(90deg, rgba(150,210,245,0.6), #3c8cdc, #0e3ca0)" }} />
              <div className="ticks"><span>0.5</span><span>0.75</span><span>1.0</span></div>
              {layers.uncertainty.on && (
                <>
                  <b style={{ display: "block", marginTop: 8 }}>Incertidumbre (σ)</b>
                  <div className="bar" style={{ background: "linear-gradient(90deg, rgba(253,200,120,0.3), #fdc878 20%, #961e5a)" }} />
                  <div className="ticks"><span>0</span><span>0.25</span><span>0.5</span></div>
                </>
              )}
              <div className="row">
                <span style={{ width: 14, height: 10, borderRadius: 2, background: "repeating-linear-gradient(45deg,#f1eee8 0 3px,#d6d1c8 3px 5px)", border: "1px solid #cfc8bc" }} />
                Nubes o sin datos
              </div>
            </div>
          )}
        </div>

        <aside className="side">
          <div className="card">
            <h3>Capas</h3>
            {LAYERS.slice().reverse().map((l) => (
              <div key={l.id} className={`layer ${layers[l.id].on ? "" : "off"}`}>
                <input type="checkbox" id={`ly-${l.id}`} checked={layers[l.id].on} onChange={(e) => set(l.id, { on: e.target.checked })} />
                <div className="swatch" style={{ background: l.swatch }} />
                <label htmlFor={`ly-${l.id}`} className="name">
                  {l.name}
                  <small>{l.hint}</small>
                </label>
                <input type="range" min={0} max={100} value={Math.round(layers[l.id].opacity * 100)} aria-label={`Opacidad de ${l.name}`} onChange={(e) => set(l.id, { opacity: Number(e.target.value) / 100 })} />
                <span className="pct">{Math.round(layers[l.id].opacity * 100)}%</span>
              </div>
            ))}
          </div>

          <div className="card">
            <h3>
              Escena completa <span className="muted">{scene ? `${fmtDate(scene.date, false)} · ${scene.tile}` : "—"}</span>
            </h3>
            <div className="stats">
              <div className="stat">
                <div className="lbl">Área inundada</div>
                <div className="val water">{flood?.available && !flood.is_reference ? `${num(flood.flooded_km2!)} km²` : "—"}</div>
                <div className="sub">{flood?.is_reference ? "esta es la imagen pre-evento" : flood?.available ? `frente a ${fmtDate(flood.reference_scene!.split("_")[1])}` : "requiere imagen pre-evento"}</div>
              </div>
              <div className="stat">
                <div className="lbl">Agua detectada</div>
                <div className="val">{px ? `${num(km2(px.water_valid))} km²` : "—"}</div>
                <div className="sub">píxeles válidos, incluye mar</div>
              </div>
              <div className="stat">
                <div className="lbl">Nubes</div>
                <div className="val">{px && total ? `${num((100 * px.cloud) / total)} %` : "—"}</div>
                <div className="sub">excluidas del cálculo</div>
              </div>
              <div className="stat">
                <div className="lbl">Sin datos</div>
                <div className="val">{px && total ? `${num((100 * px.nodata) / total)} %` : "—"}</div>
                <div className="sub">fuera de la órbita</div>
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Resumen del evento</h3>
            <dl className="kv">
              {event.facts.map((f) => (
                <div key={f.label} style={{ display: "contents" }}>
                  <dt>{f.label}</dt>
                  <dd>
                    {f.value}
                    <small>{f.source}</small>
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </aside>
      </div>
      <Timeline event={event} scenes={scenes} scene={scene} onScene={onScene} />
    </div>
  );
}
