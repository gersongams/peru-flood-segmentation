const MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct", "nov", "dic"];

/** "2023-03-22" → "22 mar 2023" (or "22 mar" without the year). */
export function fmtDate(iso: string, year = true): string {
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]}${year ? ` ${y}` : ""}`;
}

/** Days from a to b (ISO dates). */
export function days(a: string, b: string): number {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86_400_000);
}

/** Position of a date inside [start, end] as a CSS percentage. */
export function pos(date: string, [start, end]: [string, string]): string {
  return `${((days(start, date) / days(start, end)) * 100).toFixed(2)}%`;
}

/** Spanish number formatting with a thin space for thousands: 1 023.4 */
export function num(v: number, digits = 1): string {
  return v.toLocaleString("es-PE", { minimumFractionDigits: digits, maximumFractionDigits: digits }).replace(/,/g, " ");
}

export function mmss(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/** Area of n 10 m pixels in km². */
export const km2 = (pixels: number) => (pixels * 100) / 1e6;
