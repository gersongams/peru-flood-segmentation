import type { Scene } from "../api";
import type { FloodEvent } from "../config";
import { days, fmtDate, pos } from "../format";
import { Chevron } from "./Icons";

interface Props {
  event: FloodEvent;
  scenes: Scene[];
  scene: Scene | null;
  onScene: (s: Scene) => void;
}

const KIND_COLOR = { hazard: "#c8553d", response: "#6b6257", product: "#00857c", image: "#2563b8" };

function ticks([start, end]: [string, string]): string[] {
  const n = days(start, end);
  const step = n > 40 ? 7 : 5;
  const out: string[] = [];
  for (let d = 0; d <= n; d += step) out.push(new Date(Date.parse(start) + d * 86_400_000).toISOString().slice(0, 10));
  return out;
}

export default function Timeline({ event, scenes, scene, onScene }: Props) {
  const hazard = event.events.filter((e) => e.kind === "hazard");
  const hzStart = hazard[0]?.date;
  const hzEnd = hazard[hazard.length - 1]?.date;
  const i = scene ? scenes.findIndex((s) => s.scene === scene.scene) : -1;
  const step = (d: number) => {
    const s = scenes[i + d];
    if (s) onScene(s);
  };

  return (
    <footer className="footer">
      <div className="tl">
        <div className="tl-head">
          <h3>Línea de tiempo</h3>
          <div style={{ display: "flex", gap: 6 }}>
            <button className="iconbtn" aria-label="Escena anterior" disabled={i <= 0} onClick={() => step(-1)}><Chevron dir="left" /></button>
            <button className="iconbtn" aria-label="Escena siguiente" disabled={i < 0 || i >= scenes.length - 1} onClick={() => step(1)}><Chevron /></button>
          </div>
          <div className="spacer" />
          <span className="legend-inline"><span className="dot" style={{ width: 10, height: 10, background: "#2563b8" }} />Escena procesada</span>
          <span className="legend-inline"><span style={{ width: 0, height: 0, borderLeft: "6px solid transparent", borderRight: "6px solid transparent", borderBottom: "10px solid #c8553d" }} />Hito del evento</span>
        </div>
        <div className="track">
          {hzStart && hzEnd && <div className="hazard" style={{ left: pos(hzStart, event.range), width: `calc(${pos(hzEnd, event.range)} - ${pos(hzStart, event.range)})` }} />}
          <div className="rail" />
          {event.events.filter((e) => e.kind !== "image").map((e) => (
            <div key={e.date + e.label} className="mark" title={`${fmtDate(e.date)} · ${e.label}`} style={{ left: pos(e.date, event.range), borderBottomColor: KIND_COLOR[e.kind] }} />
          ))}
          {scenes.map((s) => (
            <button key={s.scene} className={`acq ready ${scene?.scene === s.scene ? "on" : ""}`} style={{ left: pos(s.date, event.range) }} title={`${fmtDate(s.date)} · ${s.tile}`} aria-label={`Ver ${fmtDate(s.date)}, tesela ${s.tile}`} onClick={() => onScene(s)} />
          ))}
          {scene && <div className="now" style={{ left: pos(scene.date, event.range) }}>{fmtDate(scene.date, false)}</div>}
          <div className="ticks">{ticks(event.range).map((t) => <span key={t}>{fmtDate(t, false)}</span>)}</div>
        </div>
      </div>
      <div className="events">
        <h3>Hitos del evento <span className="hint">con fuente</span></h3>
        {event.events.map((e) => (
          <div key={e.date + e.label} className="evrow">
            <span className="dot" style={{ width: 10, height: 10, background: KIND_COLOR[e.kind] }} />
            {e.url ? <a href={e.url} target="_blank" rel="noreferrer" title={e.source}>{e.label}</a> : <span title={e.source}>{e.label}</span>}
            <span className="dt">{fmtDate(e.date, false)}</span>
          </div>
        ))}
      </div>
    </footer>
  );
}
