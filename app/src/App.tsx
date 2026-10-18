import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Scene } from "./api";
import { EVENTS, type EventId } from "./config";
import { fmtDate } from "./format";
import Evidence from "./components/Evidence";
import ExportDialog from "./components/ExportDialog";
import { Calendar, Download, Logo } from "./components/Icons";
import NewAnalysis from "./components/NewAnalysis";
import Reference from "./components/Reference";
import FloodTimeline from "./components/FloodTimeline";
import Viewer from "./components/Viewer";

type Tab = "visor" | "nuevo" | "referencia" | "linea" | "evidencia";
const TABS: { id: Tab; label: string }[] = [
  { id: "visor", label: "Visor" },
  { id: "nuevo", label: "Nuevo análisis" },
  { id: "referencia", label: "Referencia" },
  { id: "linea", label: "Línea de tiempo" },
  { id: "evidencia", label: "Evidencia" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("visor");
  const [eventId, setEventId] = useState<EventId>("yaku2023");
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [exporting, setExporting] = useState(false);
  const [view, setView] = useState<[number, number, number, number] | null>(null);
  const ev = EVENTS[eventId];

  const refresh = useCallback(async () => {
    try {
      const s = await api.scenes();
      setScenes(s);
      setOnline(true);
      return s;
    } catch {
      setOnline(false);
      return [];
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const eventScenes = useMemo(
    () => scenes.filter((s) => s.date >= ev.range[0] && s.date <= ev.range[1]).sort((a, b) => a.date.localeCompare(b.date) || a.tile.localeCompare(b.tile)),
    [scenes, ev],
  );
  const scene = eventScenes.find((s) => s.scene === sceneId) ?? eventScenes.find((s) => ev.dates.includes(s.date)) ?? eventScenes[0] ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Logo /></div>
          <b>Inundaciones Perú</b>
          <small>Análisis satelital con IA</small>
        </div>
        <div className="vsep" />
        <select className="select" aria-label="Evento" value={eventId} onChange={(e) => { setEventId(e.target.value as EventId); setSceneId(null); }}>
          {(Object.keys(EVENTS) as EventId[]).map((id) => <option key={id} value={id}>{EVENTS[id].name}</option>)}
        </select>
        <span className="muted loc">{ev.location}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Calendar /> {fmtDate(ev.range[0], false)} – {fmtDate(ev.range[1])}
        </span>
        <nav className="tabs" aria-label="Secciones">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)} aria-current={tab === t.id ? "page" : undefined}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="spacer" />
        <span className="chipbtn" title="Modelo principal (configs/frozen.yaml, grupo peru/unet)">
          <span className="dot" style={{ background: online ? "#2f8a4c" : online === false ? "#d4553c" : "#9a9083" }} />
          {online === false ? "Servidor no disponible" : <>Modelo: <b>U-Net (Perú) · 3 semillas</b></>}
        </span>
        <button className="btn dark" disabled={!scene} onClick={() => setExporting(true)}><Download /> Exportar</button>
      </header>

      {tab === "visor" && (
        <Viewer event={ev} scenes={eventScenes} scene={scene} onScene={(s) => setSceneId(s.scene)} onNew={() => setTab("nuevo")} onView={setView} />
      )}
      {tab === "nuevo" && (
        <NewAnalysis
          eventId={eventId}
          onEvent={(e) => { setEventId(e); setSceneId(null); }}
          onDone={async (id) => { await refresh(); setSceneId(id); setTab("visor"); }}
        />
      )}
      {tab === "evidencia" && <Evidence />}
      {tab === "referencia" && (
        <Reference event={ev} scenes={eventScenes} scene={scene} onScene={(s) => setSceneId(s.scene)} />
      )}
      {tab === "linea" && (
        <FloodTimeline eventId={eventId} event={ev} scenes={eventScenes} scene={scene} onScene={(s) => setSceneId(s.scene)} onRefresh={refresh} />
      )}
      {exporting && scene && <ExportDialog scene={scene} view={view} onClose={() => setExporting(false)} />}
    </div>
  );
}
