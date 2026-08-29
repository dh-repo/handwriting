import { ConfidenceTier } from "../types/ocr";

export interface ConfidenceColorStyle {
  tier: ConfidenceTier;
  stroke: string;
  fill: string;
  textColor: string;
  badgeBg: string;
  badgeBorder: string;
  tailwindText: string;
}

export function getConfidenceTier(confidence: number): ConfidenceTier {
  if (confidence >= 0.90) return "high";
  if (confidence >= 0.70) return "medium";
  return "low";
}

export function getConfidenceColor(confidence: number): ConfidenceColorStyle {
  const tier = getConfidenceTier(confidence);

  switch (tier) {
    case "high":
      return {
        tier: "high",
        stroke: "#10b981", // emerald-500
        fill: "rgba(16, 185, 129, 0.15)",
        textColor: "#059669",
        badgeBg: "bg-emerald-500/10 dark:bg-emerald-950/50",
        badgeBorder: "border-emerald-500/30",
        tailwindText: "text-emerald-600 dark:text-emerald-400",
      };
    case "medium":
      return {
        tier: "medium",
        stroke: "#f59e0b", // amber-500
        fill: "rgba(245, 158, 11, 0.20)",
        textColor: "#d97706",
        badgeBg: "bg-amber-500/10 dark:bg-amber-950/50",
        badgeBorder: "border-amber-500/30",
        tailwindText: "text-amber-600 dark:text-amber-400",
      };
    case "low":
    default:
      return {
        tier: "low",
        stroke: "#f43f5e", // rose-500
        fill: "rgba(244, 63, 94, 0.25)",
        textColor: "#e11d48",
        badgeBg: "bg-rose-500/10 dark:bg-rose-950/50",
        badgeBorder: "border-rose-500/30",
        tailwindText: "text-rose-600 dark:text-rose-400",
      };
  }
}
