import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Flood, type Job, type Scene } from "../api";
import type { EventId, FloodEvent } from "../config";
import { fmtDate, mmss, num } from "../format";
import MapView, { type LayerState } from "./MapView";

interface Props {
  eventId: EventId;
  event: FloodEvent;
  scenes: Scene[];
  scene: Scene | null;
  onScene: (s: Scene) => void;
  onRefresh: () => Promise<unknown>;
}

const IDS = ["rgb", "flood", "mask"];

export default function FloodTimeline({ eventId, event, scenes, scene, onScene, onRefresh }: Props) {
  const tile = scene?.tile ?? scenes[0]?.tile ?? null;
  const tileScenes = useMemo(() => scenes.filter((s) => s.tile === tile), [scenes, tile]);
  const [floods, setFloods] = useState<Record<string, Flood>>({});
  const [mode, setMode] = useState<"flood" | "total">("flood");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all(tileScenes.map((s) => api.flood(s.scene).then((f) => [s.scene, f] as const)))
      .then((pairs) => alive && setFloods(Object.fromEntries(pairs)))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [tileScenes]);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") {
      if (job?.status === "done") onRefresh();
      return;
    }
    const t = window.setInterval(async () => setJob(await api.job(job.id)), 1000);
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job]);

  const anyFlood = Object.values(floods).find((f) => f.available);
  const record = Object.values(floods).find((f) => f.record)?.record ?? null;
  const refScene = anyFlood?.reference_scene ?? null;
  const rows = tileScenes
    .filter((s) => s.scene !== refScene && floods[s.scene]?.available)
    .map((s) => ({ date: s.date, label: fmtDate(s.date, false), flooded: floods[s.scene].flooded_km2!, water: floods[s.scene].water_km2!, valid: floods[s.scene].valid_both_share! }));
  const shown = scene && scene.tile === tile && scene.scene !== refScene ? scene : tileScenes.find((s) => s.scene !== refScene) ?? null;
  const layers: Record<string, LayerState> = { rgb: { on: true, opacity: 1 }, flood: { on: true, opacity: 1 }, mask: { on: true, opacity: 0.8 } };
  const hazard = event.events.filter((e) => e.kind === "hazard");

  const startPreevent = async () => {
    if (!tile) return;
    setError(null);
    try {
      setJob(await api.createPreevent({ event: eventId, tile }));
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="viewer" style={{ gridTemplateRows: "1fr" }}>
      <div className="main" style={{ gridTemplateColumns: "1fr 460px" }}>
        <div className="mapwrap">
          <MapView scene={shown} layers={layers} layerIds={IDS} center={event.center} zoom={event.zoom} />
          {shown && (
            <div className="ov floating datechip" style={{ left: 56 }}>
              <div className="d">{fmtDate(shown.date)}</div>
              <div className="s">{refScene ? `frente a ${fmtDate(refScene.split("_")[1])} (pre-evento)` : "sin imagen pre-evento"}</div>
            </div>
          )}
          <div className="ov floating legend">
            <div className="row" style={{ marginTop: 0 }}><span className="dot" style={{ width: 14, height: 14, borderRadius: 3, background: "#1d4f9a" }} />Área inundada: agua nueva</div>
            <div className="row"><span className="dot" style={{ width: 14, height: 14, borderRadius: 3, background: "#7fb0e0" }} />Agua ya presente antes del evento</div>
          </div>
        </div>

        <aside className="side">
          <div className="card">
            <h3>Evolución del agua <span className="muted">tesela {tile ?? "—"}</span></h3>
            <div className="seg" style={{ marginBottom: 10 }}>
              <button className={mode === "flood" ? "on" : ""} onClick={() => setMode("flood")}>Área inundada</button>
              <button className={mode === "total" ? "on" : ""} onClick={() => setMode("total")}>Agua detectada (incluye mar)</button>
            </div>
            {!refScene && (
              <div className="note" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <span><b>Falta la imagen pre-evento.</b> El área inundada se mide contra la imagen más despejada de la tesela en los 60 días anteriores al inicio del evento. Se calcula una sola vez.</span>
                {!job && <button className="btn dark" onClick={startPreevent} disabled={!tile}>Buscar y procesar imagen pre-evento</button>}
                {job && (
                  <span className="mono" style={{ fontSize: 12 }}>
                    {job.status === "error" ? job.error : job.status === "done" ? "Listo." : `${mmss(job.elapsed ?? 0)} · ${Object.entries(job.stages).find(([, s]) => s.status === "running")?.[1].detail || "en cola"}`}
                  </span>
                )}
              </div>
            )}
            {error && <div className="error">{error}</div>}
            {rows.length > 0 && (
              <div style={{ height: 220 }}>
                <ResponsiveContainer>
                  <BarChart data={rows} margin={{ top: 16, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#e0d8cb" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} />
                    <YAxis tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} width={48} unit="" />
                    {hazard.length > 1 && <ReferenceArea x1={fmtDate(hazard[0].date, false)} x2={fmtDate(hazard[hazard.length - 1].date, false)} fill="#c8553d" fillOpacity={0.08} />}
                    <Tooltip formatter={(v: number) => `${num(v)} km²`} labelFormatter={(l) => String(l)} />
                    <Bar dataKey={mode === "flood" ? "flooded" : "water"} name={mode === "flood" ? "Área inundada" : "Agua detectada"} fill={mode === "flood" ? "#1d4f9a" : "#7fb0e0"} radius={[5, 5, 0, 0]} label={{ position: "top", fontSize: 11, fontFamily: "IBM Plex Mono", formatter: (v: number) => num(v, 0) }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {rows.length > 0 && (
            <div className="card">
              <h3>Por fecha <span className="muted">km²</span></h3>
              <table className="evtable" style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                <thead>
                  <tr style={{ color: "var(--ink-2)", textAlign: "left" }}>
                    <th style={{ fontWeight: 500, padding: "4px 0" }}>Fecha</th>
                    <th style={{ fontWeight: 500, textAlign: "right" }}>Área inundada</th>
                    <th style={{ fontWeight: 500, textAlign: "right" }}>Agua detectada*</th>
                    <th style={{ fontWeight: 500, textAlign: "right" }}>Válidos en ambas</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.date} onClick={() => { const s = tileScenes.find((x) => x.date === r.date); if (s) onScene(s); }} style={{ cursor: "pointer", background: shown?.date === r.date ? "#eef4fb" : undefined }}>
                      <td style={{ padding: "6px 0", borderTop: "1px solid var(--line-2)" }}>{fmtDate(r.date)}</td>
                      <td className="mono" style={{ textAlign: "right", borderTop: "1px solid var(--line-2)", color: "var(--water-dark)" }}>{num(r.flooded)}</td>
                      <td className="mono" style={{ textAlign: "right", borderTop: "1px solid var(--line-2)" }}>{num(r.water)}</td>
                      <td className="mono" style={{ textAlign: "right", borderTop: "1px solid var(--line-2)" }}>{num(100 * r.valid, 0)} %</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="hint" style={{ marginTop: 8 }}>* Incluye el mar y el agua permanente dentro de la tesela.</div>
              {record && <div className="hint" style={{ marginTop: 4 }}>Imagen pre-evento: {fmtDate(record.chosen)} (nube espesa {num(100 * (record.ranking.find((x) => x.date === record.chosen)?.thick_cloud ?? 0), 2)} %, entre {record.ranking.length} fechas del {fmtDate(record.window[0])} al {fmtDate(record.window[1])}).</div>}
            </div>
          )}

          <div className="card">
            <h3>Qué se mide</h3>
            <div className="note" style={{ lineHeight: 1.5 }}>
              <b>Área inundada</b>: píxeles que pasan de seco a agua entre la imagen pre-evento y la fecha, contando solo píxeles válidos (sin nubes ni falta de datos) en ambas. <b>Agua detectada</b>: toda el agua de la fecha, incluidos ríos, lagunas y agua anterior al evento.
              <div className="mono" style={{ marginTop: 8, fontSize: 12 }}>A = Área((W<sub>t</sub> \ W<sub>0</sub>) ∩ V<sub>t</sub> ∩ V<sub>0</sub>), agua: p &gt; 0.5</div>
            </div>
            <p className="hint" style={{ margin: "8px 0 0" }}>El agua que ya existía y se expandió no cuenta como inundación si ese píxel estaba mojado antes del evento.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
