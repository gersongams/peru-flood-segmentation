import { useEffect, useRef, useState } from "react";
import type { Scene } from "../api";
import { fmtDate } from "../format";
import { Close, Download } from "./Icons";

interface Props {
  scene: Scene;
  view: [number, number, number, number] | null;
  onClose: () => void;
}

const ITEMS = [
  { id: "probability", name: "Probabilidad de agua", detail: "Media del ensamble, 0–1, 10 m", fmt: "GeoTIFF" },
  { id: "uncertainty", name: "Incertidumbre", detail: "Desviación estándar entre semillas", fmt: "GeoTIFF" },
  { id: "flood", name: "Mapa de inundación", detail: "Agua nueva frente a la imagen pre-evento (p > 0.5)", fmt: "GeoTIFF" },
  { id: "mask", name: "Nubes y sin datos", detail: "Máscara de píxeles no válidos", fmt: "GeoTIFF" },
  { id: "polygons", name: "Polígonos de inundación", detail: "Vectorizado del mapa de inundación, WGS 84", fmt: "GeoJSON" },
];

export default function ExportDialog({ scene, view, onClose }: Props) {
  const [on, setOn] = useState<Record<string, boolean>>({ probability: true, uncertainty: true, flood: true, mask: false, polygons: true });
  const [extent, setExtent] = useState<"tile" | "view">("tile");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    ref.current?.showModal();
  }, []);

  const download = async () => {
    setBusy(true);
    setError(null);
    const q = new URLSearchParams({ layers: ITEMS.filter((i) => on[i.id]).map((i) => i.id).join(",") });
    if (extent === "view" && view) q.set("bbox", view.join(","));
    try {
      const r = await fetch(`/api/export/${scene.scene}?${q}`);
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${scene.scene}${extent === "view" ? "_vista" : ""}.zip`;
      a.click();
      URL.revokeObjectURL(a.href);
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const count = ITEMS.filter((i) => on[i.id]).length;
  return (
    <dialog ref={ref} className="dialog" onClose={onClose} aria-labelledby="exp-title">
      <div className="dlg-head">
        <div>
          <h2 id="exp-title">Exportar resultados</h2>
          <div className="muted">Tesela {scene.tile} · {fmtDate(scene.date)} · U-Net entrenada con Perú, 3 semillas</div>
        </div>
        <button className="iconbtn" aria-label="Cerrar" onClick={onClose} style={{ border: 0, background: "transparent" }}><Close /></button>
      </div>
      <div className="dlg-body">
        <fieldset>
          <legend>Capas</legend>
          <div className="list" style={{ maxHeight: "none" }}>
            {ITEMS.map((i) => (
              <label key={i.id} style={{ gridTemplateColumns: "18px minmax(0,1fr) auto" }}>
                <input type="checkbox" checked={on[i.id]} onChange={(e) => setOn({ ...on, [i.id]: e.target.checked })} />
                <span>{i.name}<small>{i.detail}</small></span>
                <span className="pill" style={i.fmt === "GeoJSON" ? { background: "#e2f1ef", color: "#00635c" } : { background: "#e4ecf7", color: "#1d4f9a" }}>{i.fmt}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
          <fieldset style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <legend>Extensión</legend>
            <label style={{ display: "flex", gap: 8 }}><input type="radio" name="ext" checked={extent === "tile"} onChange={() => setExtent("tile")} />Tesela completa (~110 × 110 km)</label>
            <label style={{ display: "flex", gap: 8, opacity: view ? 1 : 0.5 }}><input type="radio" name="ext" disabled={!view} checked={extent === "view"} onChange={() => setExtent("view")} />Vista actual del mapa</label>
          </fieldset>
          <fieldset style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <legend>Sistema de referencia</legend>
            <span className="muted">Rásteres: UTM 17S (EPSG:32717)</span>
            <span className="muted">GeoJSON: WGS 84 (EPSG:4326)</span>
          </fieldset>
        </div>
        <div className="note">
          La descarga incluye <span className="mono">metadatos.json</span>: modelo, sha256 de los pesos, versión del sistema, producto Sentinel-2 e imagen pre-evento. Sin imagen pre-evento, el mapa de inundación se reemplaza por el agua detectada y así se indica.
        </div>
        {error && <div className="error">{error}</div>}
      </div>
      <div className="dlg-foot">
        <span className="muted">{count} archivos + metadatos · ZIP</span>
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn" onClick={onClose} style={{ minHeight: 44 }}>Cancelar</button>
          <button className="btn primary" onClick={download} disabled={busy || count === 0} style={{ minHeight: 44 }}>
            <Download size={16} /> {busy ? "Preparando…" : "Descargar"}
          </button>
        </div>
      </div>
    </dialog>
  );
}
