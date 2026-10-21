import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Scene } from "./api";
import { EVENTS, type EventId } from "./config";
import { fmtDate } from "./format";
import Evidence from "./components/Evidence";
import Home from "./components/Home";
import ExportDialog from "./components/ExportDialog";
import { Calendar, Download, Logo } from "./components/Icons";
import NewAnalysis from "./components/NewAnalysis";
import Reference from "./components/Reference";
import FloodTimeline from "./components/FloodTimeline";
import Viewer from "./components/Viewer";

type Tab = "inicio" | "visor" | "nuevo" | "referencia" | "linea" | "evidencia";
const TABS: { id: Tab; label: string }[] = [
  { id: "inicio", label: "Inicio" },
  { id: "visor", label: "Visor" },
  { id: "nuevo", label: "Nuevo análisis" },
  { id: "referencia", label: "Referencia" },
  { id: "linea", label: "Línea de tiempo" },
  { id: "evidencia", label: "Evidencia" },
];

// One line under the top bar: what each screen is for.
const INTRO: Partial<Record<Tab, [string, string]>> = {
  visor: ["Visor:", "agua detectada por la IA (azul) sobre la imagen satelital. Pasa el cursor para ver la probabilidad y la incertidumbre de cada píxel."],
  nuevo: ["Nuevo análisis:", "elige tesela y fecha; el sistema descarga la imagen de Copernicus y genera el mapa de agua en unos minutos."],
  referencia: ["Referencia:", "compara el mapa de la IA con el mapa experto (UNOSAT en 2023, Copernicus EMS en 2017): aciertos, agua falsa y agua omitida."],
  linea: ["Línea de tiempo:", "área inundada por fecha = agua nueva respecto de una imagen anterior al evento."],
  evidencia: ["Evidencia:", "qué tan bien funciona cada modelo en datos de prueba y qué límites tiene."],
};

export default function App() {
  const [tab, setTab] = useState<Tab>("inicio");
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
          {online === false ? "Servidor no disponible" : <>Modelo: <b>U-Net (Perú)</b></>}
        </span>
        {(tab === "visor" || tab === "linea" || tab === "referencia") && <button className="btn dark" disabled={!scene} onClick={() => setExporting(true)}><Download /> Exportar</button>}
      </header>

      {INTRO[tab] && (
        <div className="intro" role="note">
          <b>{INTRO[tab]![0]}</b> {INTRO[tab]![1]}
        </div>
      )}
      {tab === "inicio" && <Home event={ev} scenes={eventScenes} go={setTab} />}
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
