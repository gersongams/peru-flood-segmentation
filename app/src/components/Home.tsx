import type { Scene } from "../api";
import type { FloodEvent } from "../config";
import { fmtDate, num } from "../format";
import { Warn } from "./Icons";

type Tab = "visor" | "nuevo" | "referencia" | "linea" | "evidencia";

interface Props {
  event: FloodEvent;
  scenes: Scene[];
  go: (t: Tab) => void;
}

const STEPS: { n: number; tab: Tab; title: string; text: string; action: string }[] = [
  { n: 1, tab: "nuevo", title: "Elige una zona y una fecha", text: "Seleccionas el evento, una tesela de ~110 × 110 km y una fecha con imagen Sentinel-2.", action: "Nuevo análisis" },
  { n: 2, tab: "nuevo", title: "El sistema procesa la imagen", text: "Descarga la imagen de Copernicus y un modelo de IA marca, píxel por píxel (10 m), dónde hay agua. Tarda unos minutos.", action: "Ver el avance" },
  { n: 3, tab: "visor", title: "Mira el mapa de agua", text: "El agua aparece en azul sobre la imagen. La capa de incertidumbre muestra dónde el modelo duda; las nubes se marcan aparte.", action: "Abrir el visor" },
  { n: 4, tab: "linea", title: "Mide el área inundada", text: "Compara con una imagen de antes del evento: el agua nueva es el área inundada, en km², fecha por fecha.", action: "Línea de tiempo" },
  { n: 5, tab: "referencia", title: "Contrasta con mapas expertos", text: "Superpone el mapa de UNOSAT y muestra dónde coincide la IA y dónde se equivoca.", action: "Referencia" },
  { n: 6, tab: "visor", title: "Descarga los resultados", text: "Exporta los mapas como GeoTIFF y los polígonos de inundación como GeoJSON para usarlos en QGIS.", action: "Exportar desde el visor" },
];

export default function Home({ event, scenes, go }: Props) {
  return (
    <div className="page">
      <div className="doc" style={{ maxWidth: 1180 }}>
        <section className="panel hero" style={{ flex: "1 1 100%" }}>
          <div className="kicker">Sistema de análisis satelital de inundaciones · Perú</div>
          <h1>¿Dónde se inundó y cuánto?</h1>
          <p className="lead">
            Esta aplicación detecta el agua en imágenes del satélite Sentinel-2 con un modelo de inteligencia artificial entrenado con inundaciones
            del Perú. A partir de una zona y una fecha, entrega el <b>mapa de agua</b>, la <b>incertidumbre</b> del modelo, el <b>área inundada</b> frente
            a una imagen previa al evento y su <b>comparación con mapas expertos</b>.
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 14 }}>
            <button className="btn primary" style={{ minHeight: 44, padding: "0 18px" }} onClick={() => go(scenes.length ? "visor" : "nuevo")}>
              {scenes.length ? "Ver el mapa del evento" : "Empezar un análisis"}
            </button>
            <button className="btn" style={{ minHeight: 44 }} onClick={() => go("nuevo")}>Procesar otra zona o fecha</button>
          </div>
          {scenes.length > 0 && (
            <p className="hint" style={{ marginTop: 12 }}>
              Ya procesado para {event.name}: {scenes.map((s) => `${s.tile} ${fmtDate(s.date, false)}`).join(" · ")}.
            </p>
          )}
        </section>

        <section style={{ flex: "1 1 100%" }}>
          <h2 className="section-title">Cómo se usa</h2>
          <ol className="steps">
            {STEPS.map((s) => (
              <li key={s.n} className="panel">
                <span className="step">{s.n}</span>
                <b>{s.title}</b>
                <p>{s.text}</p>
                <button className="linkbtn" onClick={() => go(s.tab)}>{s.action} →</button>
              </li>
            ))}
          </ol>
        </section>

        <section className="panel" style={{ flex: "2 1 520px" }}>
          <h2>Qué hay detrás</h2>
          <dl className="kv" style={{ gridTemplateColumns: "auto 1fr" }}>
            <dt>Imágenes</dt><dd className="left">Sentinel-2 (Copernicus), 13 bandas, píxel de 10 m, una pasada cada ~5 días</dd>
            <dt>Modelo</dt><dd className="left">U-Net entrenada con mapas de inundación del Perú (UNOSAT 2023, Copernicus EMS 2017); 3 versiones que votan juntas</dd>
            <dt>Agua</dt><dd className="left">píxel con probabilidad de agua mayor que 0.5</dd>
            <dt>Incertidumbre</dt><dd className="left">cuánto discrepan las 3 versiones del modelo</dd>
            <dt>Área inundada</dt><dd className="left">agua nueva respecto de la imagen más despejada de los 60 días previos al evento</dd>
            <dt>Eventos</dt><dd className="left">Ciclón Yaku 2023 (Lambayeque y Piura) y Niño Costero 2017 (Bajo Piura)</dd>
          </dl>
          <p className="hint" style={{ marginTop: 10 }}>
            Precisión medida una sola vez en datos de prueba: IoU {num(0.828, 3)} en 2023 (pestaña <button className="linkbtn" onClick={() => go("evidencia")}>Evidencia</button>).
          </p>
        </section>

        <section className="panel" style={{ flex: "1 1 320px" }}>
          <h2>Tenga en cuenta</h2>
          <ul className="limits">
            <li><Warn /><span>Es un <b>prototipo de investigación</b> de una tesis de maestría, no un servicio oficial de emergencias.</span></li>
            <li><Warn /><span>Las <b>nubes</b> ocultan el suelo: ahí no hay resultado.</span></li>
            <li><Warn /><span>El agua muy tenue o turbia puede pasar desapercibida.</span></li>
          </ul>
        </section>
      </div>
    </div>
  );
}
