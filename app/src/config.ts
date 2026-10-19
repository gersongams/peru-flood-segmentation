// Event, date and layer configuration. Every key event carries its source (docs/literature_thesis.md).

export type EventId = "yaku2023" | "nino2017";

export interface KeyEvent {
  date: string; // ISO date
  label: string;
  kind: "hazard" | "response" | "product" | "image";
  source: string;
  url?: string;
}

export interface FloodEvent {
  id: EventId;
  name: string;
  short: string;
  location: string;
  range: [string, string]; // timeline extent (ISO dates)
  dates: string[]; // Sentinel-2 dates with an expert reference map
  center: [number, number];
  zoom: number;
  reference: string;
  facts: { label: string; value: string; source: string }[];
  events: KeyEvent[];
}

export const EVENTS: Record<EventId, FloodEvent> = {
  yaku2023: {
    id: "yaku2023",
    name: "Ciclón Yaku 2023",
    short: "Yaku 2023",
    location: "Lambayeque y Piura, Perú",
    range: ["2023-03-01", "2023-04-15"],
    dates: ["2023-03-22", "2023-04-01"],
    center: [-80.55, -6.0],
    zoom: 8.6,
    reference: "UNOSAT FL20230321PER (preliminar, sin validación de campo)",
    facts: [
      { label: "Fechas del evento", value: "4–20 mar 2023", source: "INDECI, Compendio estadístico 2024, Tabla 6, p. 86" },
      { label: "Damnificados (Yaku)", value: "107 780", source: "INDECI, Compendio estadístico 2024, Tabla 6, p. 86" },
      { label: "Fallecidos (Yaku)", value: "44", source: "INDECI, Compendio estadístico 2024, Tabla 6, p. 86" },
      { label: "Inundación (UNOSAT, 22 mar)", value: "985.9 km²", source: "UNOSAT FL20230321PER, tabla en HDX" },
    ],
    events: [
      { date: "2023-03-04", label: "Inicio de lluvias (Yaku)", kind: "hazard", source: "INDECI, Compendio 2024, Tabla 6" },
      { date: "2023-03-14", label: "Carta Internacional, activación 811", kind: "response", source: "International Charter", url: "https://disasterscharter.org/activations/flood-large-in-peru-activation-811-" },
      { date: "2023-03-15", label: "SENAMHI: informe del ciclón Yaku", kind: "hazard", source: "SENAMHI", url: "https://sinia.minam.gob.pe/sites/default/files/sinia/archivos/public/docs/reporte_-_ciclon_yaku_f_f_.pdf" },
      { date: "2023-03-20", label: "Fin del periodo de lluvias intensas", kind: "hazard", source: "INDECI, Compendio 2024, Tabla 6" },
      { date: "2023-03-22", label: "Imagen Sentinel-2 · mapa UNOSAT", kind: "image", source: "Copernicus Sentinel-2; UNOSAT" },
      { date: "2023-03-26", label: "Emergencia nacional (DS 043-2023-PCM)", kind: "response", source: "El Peruano", url: "https://busquedas.elperuano.pe/dispositivo/NL/2181041-2" },
      { date: "2023-04-01", label: "Imagen Sentinel-2 · mapa UNOSAT", kind: "image", source: "Copernicus Sentinel-2; UNOSAT" },
    ],
  },
  nino2017: {
    id: "nino2017",
    name: "Niño Costero 2017",
    short: "Niño Costero 2017",
    location: "Bajo Piura y Sechura, Perú",
    range: ["2017-03-15", "2017-04-15"],
    dates: ["2017-04-07"],
    center: [-80.62, -5.45],
    zoom: 9.2,
    reference: "Copernicus EMS EMSR199 (delineación sobre radar, 23–30 mar 2017)",
    facts: [
      { label: "Damnificados (nacional)", value: "285 453", source: "INDECI, Compendio estadístico 2017, pp. 13–16" },
      { label: "Fallecidos (nacional)", value: "138", source: "INDECI, Compendio estadístico 2017, pp. 13–16" },
      { label: "Pérdidas estimadas", value: "US$ 3.1 mil millones", source: "French et al. (2020), IJDRR 47, Tabla 1" },
      { label: "Caudal máximo río Piura", value: "3 800 m³/s (27 mar)", source: "INDECI, Compendio 2017, p. 28" },
    ],
    events: [
      { date: "2017-03-22", label: "Copernicus EMS, activación EMSR199", kind: "response", source: "Copernicus EMS", url: "https://mapping.emergency.copernicus.eu/activations/EMSR199/" },
      { date: "2017-03-27", label: "Máxima avenida del río Piura", kind: "hazard", source: "INDECI, Compendio 2017, p. 28" },
      { date: "2017-03-31", label: "Carta Internacional, activación 523", kind: "response", source: "International Charter", url: "https://disasterscharter.org/activations/flood-in-peru-call-602-" },
      { date: "2017-04-07", label: "Imagen Sentinel-2 (primera fecha despejada)", kind: "image", source: "Copernicus Sentinel-2" },
    ],
  },
};

export type LayerId = "rgb" | "probability" | "uncertainty" | "mask" | "compare";

export interface LayerDef {
  id: LayerId;
  name: string;
  hint: string;
  swatch: string;
  defaultOn: boolean;
  defaultOpacity: number;
}

// Drawn bottom to top.
export const LAYERS: LayerDef[] = [
  { id: "rgb", name: "Imagen Sentinel-2", hint: "Color verdadero de la fecha", swatch: "linear-gradient(135deg,#c9b48a,#7d8c6a)", defaultOn: true, defaultOpacity: 1 },
  { id: "probability", name: "Agua detectada por IA", hint: "U-Net entrenada con Perú · 3 semillas", swatch: "linear-gradient(135deg,#96d2f5,#0e3ca0)", defaultOn: true, defaultOpacity: 0.9 },
  { id: "uncertainty", name: "Incertidumbre", hint: "Desacuerdo entre semillas (σ)", swatch: "linear-gradient(135deg,#fdc878,#961e5a)", defaultOn: false, defaultOpacity: 0.85 },
  { id: "mask", name: "Nubes (sin resultado)", hint: "Bajo la nube el satélite no ve el suelo", swatch: "#e6e1d8", defaultOn: true, defaultOpacity: 0.5 },
];

// Optional comparison layer: the label-free model, only for scenes processed with it.
export const COMPARE_LAYER: LayerDef = { id: "compare", name: "IA sin etiquetas peruanas", hint: "U-Net trHM · comparación, no es el modelo principal", swatch: "linear-gradient(135deg,#c4b5f0,#4c28a0)", defaultOn: false, defaultOpacity: 0.85 };

// MGRS tiles over each event's reference extent, with their share of UNOSAT's analysed extent
// (docs/system_evaluation_protocol.md §2).
export const TILES: Record<EventId, { id: string; share?: string }[]> = {
  yaku2023: [
    { id: "17MNP", share: "49.3 %" }, { id: "17MNQ", share: "18.7 %" }, { id: "17MPN", share: "17.8 %" },
    { id: "17MNN", share: "6.7 %" }, { id: "17MPP", share: "4.9 %" }, { id: "17MPQ", share: "2.6 %" },
  ],
  nino2017: [{ id: "17MNP" }, { id: "17MNQ" }],
};
