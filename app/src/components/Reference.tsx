import { useEffect, useState } from "react";
import { api, type Metrics, type Scene } from "../api";
import type { FloodEvent } from "../config";
import { fmtDate, num } from "../format";
import MapView, { type LayerState } from "./MapView";

interface Props {
  event: FloodEvent;
  scenes: Scene[];
  scene: Scene | null;
  onScene: (s: Scene) => void;
}

type Scope = "view" | "tile" | "test";
const SCOPES: { id: Scope; label: string; note: string }[] = [
  { id: "view", label: "Vista actual", note: "Área visible del mapa, dentro de la extensión que analizó UNOSAT." },
  { id: "tile", label: "Tesela completa", note: "Toda la extensión analizada dentro de la tesela." },
  { id: "test", label: "Recortes de prueba", note: "Solo píxeles de los recortes de peru-test: zonas que el modelo no vio al entrenar ni al seleccionarse." },
];
const IDS = ["rgb", "reference", "errors", "mask"];

export default function Reference({ event, scenes, scene, onScene }: Props) {
  const [scope, setScope] = useState<Scope>("view");
  const [view, setView] = useState<[number, number, number, number] | null>(null);
  const [m, setM] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [show, setShow] = useState<"errors" | "reference">("errors");
  const withRef = scenes.filter((s) => event.dates.includes(s.date));

  useEffect(() => {
    if (!scene) return;
    let alive = true;
    setLoading(true);
    setError(null);
    const t = window.setTimeout(() => {
      api.metrics(scene.scene, scope === "view" ? view ?? undefined : undefined, scope === "test" ? "test" : undefined)
        .then((r) => alive && setM(r))
        .catch((e) => alive && setError(String(e)))
        .finally(() => alive && setLoading(false));
    }, 250);
    return () => {
      alive = false;
      window.clearTimeout(t);
    };
  }, [scene, scope, view]);

  const layers: Record<string, LayerState> = {
    rgb: { on: true, opacity: 1 },
    reference: { on: show === "reference", opacity: 1 },
    errors: { on: show === "errors", opacity: 0.95 },
    mask: { on: true, opacity: 0.8 },
  };
  const tp = m?.tp_km2 ?? 0, fp = m?.fp_km2 ?? 0, fn = m?.fn_km2 ?? 0;
  const f3 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3));

  return (
    <div className="viewer" style={{ gridTemplateRows: "1fr" }}>
      <div className="main">
        <div className="mapwrap">
          <MapView scene={scene} layers={layers} layerIds={IDS} center={event.center} zoom={event.zoom} onView={setView} />
          <div className="ov floating" style={{ top: 14, left: 56, padding: 4 }}>
            <div className="seg" style={{ border: 0 }}>
              <button className={show === "errors" ? "on" : ""} onClick={() => setShow("errors")}>Errores de la IA</button>
              <button className={show === "reference" ? "on" : ""} onClick={() => setShow("reference")}>Referencia UNOSAT</button>
            </div>
          </div>
          <div className="ov floating legend">
            {show === "errors" ? (
              <>
                <div className="row" style={{ marginTop: 0 }}><span className="dot" style={{ width: 14, height: 14, borderRadius: 3, background: "#2563b8" }} />Agua en ambos (acierto)</div>
                <div className="row"><span className="dot" style={{ width: 14, height: 14, borderRadius: 3, background: "#d4553c" }} />Agua falsa: IA sí, referencia no</div>
                <div className="row"><span className="dot" style={{ width: 14, height: 14, borderRadius: 3, background: "#e9a21b" }} />Agua omitida: referencia sí, IA no</div>
              </>
            ) : (
              <div className="row" style={{ marginTop: 0 }}><span className="dot" style={{ width: 14, height: 14, borderRadius: 3, background: "rgba(0,164,154,0.6)" }} />Agua según UNOSAT (inundación + agua previa)</div>
            )}
          </div>
        </div>

        <aside className="side">
          <div className="card">
            <h3>Concordancia con la referencia</h3>
            {withRef.length === 0 && <div className="note">Procesa una fecha con mapa de referencia ({event.dates.map((d) => fmtDate(d)).join(", ")}) para comparar.</div>}
            {withRef.length > 0 && (
              <>
                <label className="hint" htmlFor="ref-scene">Escena</label>
                <select id="ref-scene" className="select" style={{ width: "100%", border: "1px solid var(--line)", background: "#fff", marginBottom: 10 }} value={scene?.scene ?? ""} onChange={(e) => { const s = withRef.find((x) => x.scene === e.target.value); if (s) onScene(s); }}>
                  {withRef.map((s) => <option key={s.scene} value={s.scene}>{fmtDate(s.date)} · {s.tile}</option>)}
                </select>
                <p className="muted" style={{ margin: "0 0 10px", lineHeight: 1.45 }}>Agua de la IA (p &gt; 0.5) frente al mapa experto. Solo píxeles válidos en ambos: sin nubes ni falta de datos.</p>
                <div className="seg" role="group" aria-label="Ámbito del cálculo" style={{ width: "100%" }}>
                  {SCOPES.map((s) => <button key={s.id} style={{ flex: 1 }} className={scope === s.id ? "on" : ""} onClick={() => setScope(s.id)}>{s.label}</button>)}
                </div>
                <div className="hint" style={{ marginTop: 6 }}>{SCOPES.find((s) => s.id === scope)!.note}</div>
              </>
            )}
          </div>

          {scene && event.dates.includes(scene.date) && (
            <>
              <div className="card" style={{ opacity: loading ? 0.6 : 1 }}>
                {error && <div className="error">{error}</div>}
                <div className="stats" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
                  <div className="stat"><div className="lbl">IoU</div><div className="val water">{f3(m?.iou)}</div><div className="sub">intersección / unión</div></div>
                  <div className="stat"><div className="lbl">Precisión</div><div className="val">{f3(m?.precision)}</div><div className="sub">agua IA que es real</div></div>
                  <div className="stat"><div className="lbl">Exhaustividad</div><div className="val">{f3(m?.recall)}</div><div className="sub">agua real hallada</div></div>
                </div>
              </div>
              <div className="card">
                <h3>Superficie por tipo</h3>
                <div style={{ display: "flex", height: 14, borderRadius: 7, overflow: "hidden", background: "#ebe4d8" }}>
                  <div style={{ flex: tp, background: "#2563b8" }} /><div style={{ flex: fp, background: "#d4553c" }} /><div style={{ flex: fn, background: "#e9a21b" }} />
                </div>
                <dl className="kv" style={{ marginTop: 12 }}>
                  <dt>Acierto</dt><dd className="mono">{m ? `${num(tp)} km²` : "—"}</dd>
                  <dt>Agua falsa</dt><dd className="mono">{m ? `${num(fp)} km²` : "—"}</dd>
                  <dt>Agua omitida</dt><dd className="mono">{m ? `${num(fn)} km²` : "—"}</dd>
                  <dt>Agua de referencia evaluada</dt><dd className="mono">{m ? `${num(m.reference_water_km2)} km²` : "—"}</dd>
                </dl>
              </div>
              {m && scope !== "test" && (
                <div className="card">
                  <div className="note warn">
                    <b>{Math.round(100 * m.partition_share.train)} % de estos píxeles está en zonas de entrenamiento</b> (dev {Math.round(100 * m.partition_share.dev)} %, prueba {Math.round(100 * m.partition_share.test)} %). Ahí el acuerdo es optimista; para una cifra sin ese sesgo usa «Recortes de prueba».
                  </div>
                </div>
              )}
              <div className="card">
                <div className="note">
                  <b>Cómo leer estas cifras.</b> UNOSAT es un mapa preliminar, sin validación de campo: un desacuerdo puede ser error de la IA o de la referencia. Las cifras de la tesis provienen de la evaluación congelada (pestaña Evidencia), no de esta pantalla.
                </div>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
