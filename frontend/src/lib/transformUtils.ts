import { BoundingBoxTuple, PolygonPoint } from "../types/ocr";

export interface SvgRectCoords {
  x: number;
  y: number;
  width: number;
  height: number;
}

function sanitizeCoord(val: number, fallback: number = 0): number {
  if (typeof val !== "number" || isNaN(val) || !isFinite(val)) {
    return fallback;
  }
  return val;
}

/**
 * Converts normalized bounding box [ymin, xmin, ymax, xmax] in [0, 1]
 * to pixel coordinates relative to document SVG viewBox [0, 0, width, height].
 * Normalizes inverted bounds and sanitizes NaN values.
 */
export function bboxToSvgRect(
  bbox: BoundingBoxTuple,
  docWidth: number,
  docHeight: number
): SvgRectCoords {
  const [rawYmin, rawXmin, rawYmax, rawXmax] = bbox;
  const sYmin = sanitizeCoord(rawYmin, 0);
  const sXmin = sanitizeCoord(rawXmin, 0);
  const sYmax = sanitizeCoord(rawYmax, 0);
  const sXmax = sanitizeCoord(rawXmax, 0);

  const realYmin = Math.min(sYmin, sYmax);
  const realYmax = Math.max(sYmin, sYmax);
  const realXmin = Math.min(sXmin, sXmax);
  const realXmax = Math.max(sXmin, sXmax);

  const clampedXMin = Math.max(0, Math.min(realXmin, 1));
  const clampedYMin = Math.max(0, Math.min(realYmin, 1));
  const clampedXMax = Math.max(clampedXMin, Math.min(realXmax, 1));
  const clampedYMax = Math.max(clampedYMin, Math.min(realYmax, 1));

  const validDocW = Math.max(0, sanitizeCoord(docWidth, 0));
  const validDocH = Math.max(0, sanitizeCoord(docHeight, 0));

  const x = clampedXMin * validDocW;
  const y = clampedYMin * validDocH;
  const width = Math.max(1, (clampedXMax - clampedXMin) * validDocW);
  const height = Math.max(1, (clampedYMax - clampedYMin) * validDocH);

  return { x, y, width, height };
}

/**
 * Converts normalized bounding box [ymin, xmin, ymax, xmax]
 * to a 4-point polygon [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]].
 */
export function bboxToPolygon(bbox: BoundingBoxTuple): PolygonPoint[] {
  const [rawYmin, rawXmin, rawYmax, rawXmax] = bbox;
  const sYmin = sanitizeCoord(rawYmin, 0);
  const sXmin = sanitizeCoord(rawXmin, 0);
  const sYmax = sanitizeCoord(rawYmax, 0);
  const sXmax = sanitizeCoord(rawXmax, 0);

  const realYmin = Math.min(sYmin, sYmax);
  const realYmax = Math.max(sYmin, sYmax);
  const realXmin = Math.min(sXmin, sXmax);
  const realXmax = Math.max(sXmin, sXmax);

  return [
    [realXmin, realYmin],
    [realXmax, realYmin],
    [realXmax, realYmax],
    [realXmin, realYmax],
  ];
}

/**
 * Converts polygon points to SVG "x1,y1 x2,y2 ..." point string scaled to document dimensions.
 */
export function polygonToSvgPoints(
  polygon: PolygonPoint[],
  docWidth: number,
  docHeight: number
): string {
  const validDocW = Math.max(0, sanitizeCoord(docWidth, 0));
  const validDocH = Math.max(0, sanitizeCoord(docHeight, 0));
  return polygon
    .map(([px, py]) => `${(sanitizeCoord(px, 0) * validDocW).toFixed(1)},${(sanitizeCoord(py, 0) * validDocH).toFixed(1)}`)
    .join(" ");
}

/**
 * Calculates bounding box from polygon points.
 */
export function polygonToBbox(polygon: PolygonPoint[]): BoundingBoxTuple {
  if (!polygon || polygon.length === 0) {
    return [0, 0, 0, 0];
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const [rawPx, rawPy] of polygon) {
    const px = sanitizeCoord(rawPx, 0);
    const py = sanitizeCoord(rawPy, 0);
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
  }

  if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
    return [0, 0, 0, 0];
  }

  return [
    Math.max(0, Math.min(minY, 1)),
    Math.max(0, Math.min(minX, 1)),
    Math.max(0, Math.min(maxY, 1)),
    Math.max(0, Math.min(maxX, 1)),
  ];
}

/**
 * Clamps coordinates to valid [0, 1] range and ensures min <= max.
 */
export function clampBbox(bbox: BoundingBoxTuple): BoundingBoxTuple {
  const [rawYmin, rawXmin, rawYmax, rawXmax] = bbox;
  const sYmin = sanitizeCoord(rawYmin, 0);
  const sXmin = sanitizeCoord(rawXmin, 0);
  const sYmax = sanitizeCoord(rawYmax, 0);
  const sXmax = sanitizeCoord(rawXmax, 0);

  const realYmin = Math.min(sYmin, sYmax);
  const realYmax = Math.max(sYmin, sYmax);
  const realXmin = Math.min(sXmin, sXmax);
  const realXmax = Math.max(sXmin, sXmax);

  const cYmin = Math.max(0, Math.min(realYmin, 1));
  const cXmin = Math.max(0, Math.min(realXmin, 1));
  const cYmax = Math.max(cYmin, Math.min(realYmax, 1));
  const cXmax = Math.max(cXmin, Math.min(realXmax, 1));
  return [cYmin, cXmin, cYmax, cXmax];
}
