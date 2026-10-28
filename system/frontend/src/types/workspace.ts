export type LibraryLanding = "sources" | "build" | "versions" | "summary" | "text" | "kg";

export interface ProgressSnapshot {
  progress: number;
  message: string;
  status: string;
  recordedAt: string;
}


