import { describe, it, expect, vi, beforeEach } from 'vitest';
import { extractLineCropBase64, computeCropCoordinates } from '@/lib/cropUtils';
import { BoundingBoxTuple } from '@/types/ocr';

describe('cropUtils - Line Crop Extractor', () => {
  describe('computeCropCoordinates', () => {
    it('computes pixel crop coordinates from normalized bounding box', () => {
      const bbox: BoundingBoxTuple = [0.1, 0.2, 0.3, 0.8]; // [ymin, xmin, ymax, xmax]
      const naturalWidth = 1000;
      const naturalHeight = 1000;

      const coords = computeCropCoordinates(bbox, naturalWidth, naturalHeight, 0.04);
      expect(coords).not.toBeNull();
      if (!coords) return;

      // xmin=0.2, xmax=0.8, padX=(0.8-0.2)*0.04 = 0.024 -> padded xmin = 0.176 -> 176
      // ymin=0.1, ymax=0.3, padY=(0.3-0.1)*0.04 = 0.008 -> padded ymin = 0.092 -> 92
      expect(coords.cropX).toBe(176);
      expect(coords.cropY).toBe(92);
      expect(coords.cropW).toBeGreaterThan(600);
      expect(coords.cropH).toBeGreaterThan(200);
    });

    it('clamps coordinates safely to 0 and naturalWidth/naturalHeight at edges', () => {
      const bbox: BoundingBoxTuple = [0.0, 0.0, 0.05, 0.05];
      const naturalWidth = 500;
      const naturalHeight = 500;

      const coords = computeCropCoordinates(bbox, naturalWidth, naturalHeight, 0.1);
      expect(coords).not.toBeNull();
      if (!coords) return;

      expect(coords.cropX).toBe(0);
      expect(coords.cropY).toBe(0);
      expect(coords.cropW).toBeLessThanOrEqual(500);
      expect(coords.cropH).toBeLessThanOrEqual(500);
    });

    it('normalizes inverted bounding box coordinates gracefully', () => {
      const invertedBbox: BoundingBoxTuple = [0.4, 0.9, 0.2, 0.1];
      const coords = computeCropCoordinates(invertedBbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      if (!coords) return;

      expect(coords.cropX).toBe(100);
      expect(coords.cropY).toBe(200);
      expect(coords.cropW).toBe(800);
      expect(coords.cropH).toBe(200);
    });

    it('returns null for non-finite or malformed bounding boxes', () => {
      expect(computeCropCoordinates([NaN, 0, 1, 1], 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0, Infinity, 1, 1], 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([] as unknown as BoundingBoxTuple, 1000, 1000)).toBeNull();
    });

    it('handles pixel bounding boxes where coordinates exceed 1.05', () => {
      const pixelBbox: BoundingBoxTuple = [100, 200, 300, 800];
      const coords = computeCropCoordinates(pixelBbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      if (!coords) return;

      expect(coords.cropX).toBe(200);
      expect(coords.cropY).toBe(100);
      expect(coords.cropW).toBe(600);
      expect(coords.cropH).toBe(200);
    });
  });

  describe('extractLineCropBase64', () => {
    it('returns undefined if imageUrl or bbox is omitted', async () => {
      expect(await extractLineCropBase64(undefined, [0, 0, 1, 1])).toBeUndefined();
      expect(await extractLineCropBase64('blob:test', undefined)).toBeUndefined();
    });

    it('returns a base64 PNG data URL in JSDOM / canvas mock environment', async () => {
      const result = await extractLineCropBase64('blob:mock-image-url', [0.1, 0.1, 0.2, 0.9]);
      expect(result).toBeDefined();
      expect(result).toMatch(/^data:image\/png;base64,/);
    });

    it('handles canvas getContext returning null without throwing', async () => {
      const origGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null);

      try {
        const result = await extractLineCropBase64('blob:test', [0.1, 0.1, 0.2, 0.9]);
        expect(result).toBeUndefined();
      } finally {
        HTMLCanvasElement.prototype.getContext = origGetContext;
      }
    });

    it('catches image error gracefully and does not throw unhandled exception', async () => {
      const origImage = global.Image;
      class MockBrokenImage {
        crossOrigin = '';
        width = 100;
        height = 100;
        naturalWidth = 100;
        naturalHeight = 100;
        complete = false;
        private _src = '';
        set src(val: string) {
          this._src = val;
          setTimeout(() => {
            if (this.onerror) this.onerror(new Event('error'));
          }, 5);
        }
        get src() {
          return this._src;
        }
        onload: (() => void) | null = null;
        onerror: ((ev: Event) => void) | null = null;
      }
      // @ts-expect-error Mocking Image constructor for error branch
      global.Image = MockBrokenImage;

      try {
        const result = await extractLineCropBase64('bad:url', [0.1, 0.1, 0.2, 0.9]);
        expect(result).toBeDefined();
      } finally {
        global.Image = origImage;
      }
    });
  });
});
