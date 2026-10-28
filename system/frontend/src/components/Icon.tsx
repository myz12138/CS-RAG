import type { CSSProperties } from "react";
export type IconName = "overview" | "library" | "search" | "chart" | "settings" | "tasks" | "plus" | "refresh" | "arrow" | "close" | "graph" | "file" | "upload" | "check" | "layers";
const paths: Record<IconName, string> = {
  overview: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  library: "M4 3h5v18H4z M12 3h4v18h-4z M18 4l3-1 4 17-3 1z",
  search: "M10.5 3a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15 M16 16l5 5",
  chart: "M4 3v18h17 M8 16v-5 M13 16V7 M18 16V4",
  settings: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M12 2v3 M12 19v3 M2 12h3 M19 12h3 M5 5l2 2 M17 17l2 2 M5 19l2-2 M17 7l2-2",
  tasks: "M9 5h12 M9 12h12 M9 19h12 M3 5l1 1 2-3 M3 12l1 1 2-3 M3 19l1 1 2-3",
  plus: "M12 4v16 M4 12h16", refresh: "M20 7a9 9 0 1 0 1 8 M20 2v6h-6",
  arrow: "M4 12h16 M14 6l6 6-6 6", close: "M6 6l12 12 M6 18 18 6",
  graph: "M5 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6 M19 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6 M12 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6 M8 7h8 M7 10l3 5 M17 10l-3 5",
  file: "M5 2h9l5 5v15H5z M14 2v6h5 M8 12h8 M8 16h6",
  upload: "M12 16V3 M7 8l5-5 5 5 M3 15v6h18v-6", check: "M4 12l5 5L20 6",
  layers: "M2 7l10-5 10 5-10 5z M2 12l10 5 10-5 M2 17l10 5 10-5",
};
export function Icon({ name, size = 19, style }: { name: IconName; size?: number; style?: CSSProperties }) {
  return <svg width={size} height={size} viewBox="0 0 26 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}><path d={paths[name]} /></svg>;
}
