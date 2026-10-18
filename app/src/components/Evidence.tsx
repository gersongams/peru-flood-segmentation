import { useState } from "react";
import { Warn } from "./Icons";

// Frozen one-time test evaluation, docs/phaseD_test.md (peru-test: 172 chips; peru-2017-test: 101
// chips). Water IoU, mean ± std over 3 seeds. Read from the thesis results, never recomputed here.
const PERU = "#1d4f9a";
const FREE = "#c8553d";
const ROWS = [
  { name: "U-Net", group: "Entrenada con Perú", v23: 0.828, s23: "0.828 ± 0.002", v17: 0.878, s17: "0.878", color: PERU, badge: "Principal" },
  { name: "Prithvi-EO-2.0", group: "Entrenado con Perú", v23: 0.824, s23: "0.824 ± 0.007", v17: 0.878, s17: "0.878", color: PERU },
  { name: "TerraMind", group: "Entrenado con Perú", v23: 0.815, s23: "0.815 ± 0.002", v17: 0.892, s17: "0.892", color: PERU },
  { name: "U-Net trHM", group: "Sin etiquetas: igualación de histogramas", v23: 0.798, s23: "0.798 ± 0.005", v17: 0.854, s17: "0.854 ± 0.007", color: FREE },
  { name: "Prithvi-EO-2.0", group: "Sin etiquetas: solo Sen1Floods11", v23: 0.741, s23: "0.741 ± 0.014", v17: 0.937, s17: "0.937 ± 0.006", color: FREE },
  { name: "TerraMind trFDA", group: "Sin etiquetas: FDA", v23: 0.709, s23: "0.709 ± 0.020", v17: 0.831, s17: "0.831 ± 0.006", color: FREE },
];
const LIMITS = [
  ["Prototipo de investigación.", "No reemplaza los mapas oficiales de emergencia."],
  ["Referencias preliminares.", "UNOSAT 2023 no tiene validación de campo."],
  ["Agua tenue.", "Los modelos omiten agua de bajo contraste, visible en 2017."],
  ["La incertidumbre no lo avisa todo.", "El agua de bajo contraste puede tener incertidumbre baja."],
  ["Bordes del agua.", "La calibración empeora cerca de los bordes en los datos de prueba."],
];

export default function Evidence() {
  const [ev, setEv] = useState<"2023" | "2017">("2023");
  const is23 = ev === "2023";
  return (
    <div className="page">
      <div className="doc">
        <section className="panel" style={{ flex: "999 1 640px", minWidth: 0 }}>
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "flex-end", gap: 12 }}>
            <div>
              <h1>¿Qué tan bien detecta el agua cada modelo?</h1>
              <p className="muted" style={{ margin: "4px 0 0" }}>IoU de agua en los conjuntos de prueba, evaluados una sola vez. Media ± desviación estándar de 3 semillas.</p>
            </div>
            <div className="seg" role="group" aria-label="Evento">
              <button className={is23 ? "on" : ""} onClick={() => setEv("2023")}>Yaku 2023</button>
              <button className={!is23 ? "on" : ""} onClick={() => setEv("2017")}>Niño Costero 2017</button>
            </div>
          </div>
          <div className="evrows">
            <div className="r h">
              <span>Modelo</span>
              <span className="mono" style={{ display: "flex", justifyContent: "space-between" }}><span>0.5</span><span>0.75</span><span>1.0</span></span>
              <span style={{ textAlign: "right" }}>IoU</span>
            </div>
            {ROWS.map((r) => {
              const v = is23 ? r.v23 : r.v17;
              return (
                <div className="r" key={r.name + r.group}>
                  <span>
                    <span style={{ fontWeight: 500 }}>{r.name}</span>
                    {r.badge && <span className="badge" style={{ background: "#e4ecf7", color: PERU }}>{r.badge}</span>}
                    <span className="hint" style={{ display: "block" }}>{r.group}</span>
                  </span>
                  <span className="track2"><span style={{ width: `${Math.max(0, ((v - 0.5) / 0.5) * 100)}%`, background: r.color }} /></span>
                  <span className="mono" style={{ textAlign: "right" }}>{is23 ? r.s23 : r.s17}</span>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 12 }}>
            <span className="legend-inline"><span className="dot" style={{ width: 12, height: 12, borderRadius: 3, background: PERU }} />Entrenado con etiquetas peruanas</span>
            <span className="legend-inline"><span className="dot" style={{ width: 12, height: 12, borderRadius: 3, background: FREE }} />Sin etiquetas peruanas (adaptado o de origen)</span>
          </div>
          <p className="muted" style={{ margin: "12px 0 0", fontSize: 12, lineHeight: 1.5 }}>
            {is23
              ? "En 2023 las etiquetas premian eliminar agua falsa: los modelos entrenados con Perú lideran."
              : "En 2017 el orden se invierte: la referencia solo define seco donde MNDWI ≤ 0, así que la cifra mide sobre todo exhaustividad. Ningún método gana en ambos eventos."}
          </p>
          <p className="hint" style={{ margin: "6px 0 0" }}>
            Fuente: evaluación congelada de prueba (peru-test, 172 chips; peru-2017-test, 101 chips). El modelo principal (U-Net entrenada con Perú) se eligió con estos resultados a la vista; no son una validación independiente de esa elección.
          </p>
        </section>
        <aside style={{ flex: "1 1 360px", display: "flex", flexDirection: "column", gap: 16 }}>
          <section className="panel">
            <h2>Límites conocidos</h2>
            <ul className="limits">
              {LIMITS.map(([t, d]) => (
                <li key={t}><Warn /><span><b>{t}</b> <span style={{ color: "#4f473f" }}>{d}</span></span></li>
              ))}
            </ul>
          </section>
          <section className="panel">
            <h2>Cómo se genera el mapa</h2>
            <dl className="kv" style={{ gridTemplateColumns: "auto 1fr" }}>
              <dt>Imagen</dt><dd style={{ textAlign: "left", fontWeight: 400 }}>Sentinel-2 L1C, Copernicus Data Space</dd>
              <dt>Modelo</dt><dd style={{ textAlign: "left", fontWeight: 400 }}>U-Net entrenada con Perú, 3 semillas, pesos congelados</dd>
              <dt>Agua</dt><dd style={{ textAlign: "left", fontWeight: 400 }}>probabilidad media &gt; 0.5</dd>
              <dt>Incertidumbre</dt><dd style={{ textAlign: "left", fontWeight: 400 }}>desviación estándar entre semillas</dd>
            </dl>
          </section>
        </aside>
      </div>
    </div>
  );
}
