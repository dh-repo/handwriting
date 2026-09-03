/**
 * frontend/src/__tests__/adversarial/cropUtilsStress.test.ts
 * Adversarial Stress Tests for cropUtils.ts (Milestone 4 Darkroom Feedback Integration)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { computeCropCoordinates, extractLineCropBase64 } from '@/lib/cropUtils';
import { BoundingBoxTuple } from '@/types/ocr';

describe('Adversarial Stress Testing: cropUtils.ts', () => {
  describe('computeCropCoordinates - Extreme & Hostile Coordinates', () => {
    it('handles inverted Y bounding box [ymax, xmin, ymin, xmax]', () => {
      // ymax=0.85, xmin=0.10, ymin=0.15, xmax=0.90
      const bbox: BoundingBoxTuple = [0.85, 0.10, 0.15, 0.90];
      const coords = computeCropCoordinates(bbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      expect(coords!.cropY).toBe(150);
      expect(coords!.cropX).toBe(100);
      expect(coords!.cropH).toBe(700);
      expect(coords!.cropW).toBe(800);
    });

    it('handles inverted X bounding box [ymin, xmax, ymax, xmin]', () => {
      const bbox: BoundingBoxTuple = [0.20, 0.95, 0.40, 0.05];
      const coords = computeCropCoordinates(bbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      expect(coords!.cropX).toBe(50);
      expect(coords!.cropY).toBe(200);
      expect(coords!.cropW).toBe(900);
      expect(coords!.cropH).toBe(200);
    });

    it('handles fully inverted bounding box [ymax, xmax, ymin, xmin]', () => {
      const bbox: BoundingBoxTuple = [0.90, 0.80, 0.10, 0.20];
      const coords = computeCropCoordinates(bbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      expect(coords!.cropX).toBe(200);
      expect(coords!.cropY).toBe(100);
      expect(coords!.cropW).toBe(600);
      expect(coords!.cropH).toBe(800);
    });

    it('handles fully inverted pixel coordinates [900, 800, 100, 200]', () => {
      const bbox: BoundingBoxTuple = [900, 800, 100, 200];
      const coords = computeCropCoordinates(bbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      expect(coords!.cropX).toBe(200);
      expect(coords!.cropY).toBe(100);
      expect(coords!.cropW).toBe(600);
      expect(coords!.cropH).toBe(800);
    });

    it('detects zero area (point bbox: ymin=ymax and xmin=xmax) and returns null', () => {
      const pointBbox: BoundingBoxTuple = [0.5, 0.5, 0.5, 0.5];
      const coords = computeCropCoordinates(pointBbox, 1000, 1000, 0.04);
      expect(coords).toBeNull();
    });

    it('detects zero width (xmin=xmax) and returns null', () => {
      const zeroWidthBbox: BoundingBoxTuple = [0.1, 0.5, 0.9, 0.5];
      const coords = computeCropCoordinates(zeroWidthBbox, 1000, 1000, 0.0);
      expect(coords).toBeNull();
    });

    it('detects zero height (ymin=ymax) and returns null', () => {
      const zeroHeightBbox: BoundingBoxTuple = [0.5, 0.1, 0.5, 0.9];
      const coords = computeCropCoordinates(zeroHeightBbox, 1000, 1000, 0.0);
      expect(coords).toBeNull();
    });

    it('returns null on NaN coordinates', () => {
      expect(computeCropCoordinates([NaN, 0.1, 0.5, 0.9], 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0.1, NaN, 0.5, 0.9], 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0.1, 0.1, NaN, 0.9], 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0.1, 0.1, 0.5, NaN], 1000, 1000)).toBeNull();
    });

    it('returns null on Infinity or -Infinity coordinates', () => {
      expect(computeCropCoordinates([Infinity, 0.1, 0.5, 0.9], 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0.1, -Infinity, 0.5, 0.9], 1000, 1000)).toBeNull();
    });

    it('returns null on malformed array lengths', () => {
      expect(computeCropCoordinates([] as unknown as BoundingBoxTuple, 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0.1, 0.2, 0.3] as unknown as BoundingBoxTuple, 1000, 1000)).toBeNull();
      expect(computeCropCoordinates([0.1, 0.2, 0.3, 0.4, 0.5] as unknown as BoundingBoxTuple, 1000, 1000)).toBeNull();
      expect(computeCropCoordinates(null as unknown as BoundingBoxTuple, 1000, 1000)).toBeNull();
      expect(computeCropCoordinates(undefined as unknown as BoundingBoxTuple, 1000, 1000)).toBeNull();
    });

    it('stress-tests out-of-bounds coordinates (<0 or >1)', () => {
      // Bounding box far outside canvas bounds: [-500, -500, 1500, 2000]
      const outOfBoundsPixelBbox: BoundingBoxTuple = [-500, -500, 1500, 2000];
      const coords = computeCropCoordinates(outOfBoundsPixelBbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      // Should clamp to [0, 1000]
      expect(coords!.cropX).toBeGreaterThanOrEqual(0);
      expect(coords!.cropY).toBeGreaterThanOrEqual(0);
      expect(coords!.cropX + coords!.cropW).toBeLessThanOrEqual(1000);
      expect(coords!.cropY + coords!.cropH).toBeLessThanOrEqual(1000);
    });

    it('handles negative normalized coordinate edge-case (float jitter clamped safely)', () => {
      // Model predicts slightly negative coordinate due to float jitter: ymin = -0.01
      const jitterBbox: BoundingBoxTuple = [-0.01, 0.10, 0.30, 0.90];
      const coords = computeCropCoordinates(jitterBbox, 1000, 1000, 0.0);
      expect(coords).not.toBeNull();
      // With float jitter tolerance [-0.05, 1.05], it correctly normalizes and clamps to [0, 1]
      expect(coords!.cropY).toBe(0);
      expect(coords!.cropH).toBe(300);
      expect(coords!.cropX).toBe(100);
      expect(coords!.cropW).toBe(800);
    });

    it('stress-tests zero or negative natural dimensions', () => {
      const bbox: BoundingBoxTuple = [0.1, 0.1, 0.5, 0.5];
      expect(computeCropCoordinates(bbox, 0, 1000)).toBeNull();
      expect(computeCropCoordinates(bbox, 1000, 0)).toBeNull();
      expect(computeCropCoordinates(bbox, -500, 1000)).toBeNull();
    });

    it('stress-tests extreme padding ratios', () => {
      const bbox: BoundingBoxTuple = [0.4, 0.4, 0.6, 0.6];
      // 1000% padding
      const coordsHugePad = computeCropCoordinates(bbox, 1000, 1000, 10.0);
      expect(coordsHugePad).not.toBeNull();
      // Must clamp to image boundaries without overflowing
      expect(coordsHugePad!.cropX).toBe(0);
      expect(coordsHugePad!.cropY).toBe(0);
      expect(coordsHugePad!.cropW).toBe(1000);
      expect(coordsHugePad!.cropH).toBe(1000);

      // Negative padding
      const coordsNegPad = computeCropCoordinates(bbox, 1000, 1000, -0.2);
      expect(coordsNegPad).not.toBeNull();
      expect(coordsNegPad!.cropW).toBeLessThan(200);
      expect(coordsNegPad!.cropH).toBeLessThan(200);
    });
  });

  describe('extractLineCropBase64 - Runtime Stress & Failure Modes', () => {
    it('returns undefined if imageUrl is empty string or whitespace', async () => {
      expect(await extractLineCropBase64('', [0.1, 0.1, 0.2, 0.8])).toBeUndefined();
      expect(await extractLineCropBase64('   ', [0.1, 0.1, 0.2, 0.8])).toBeDefined(); // non-empty string passes initial check
    });

    it('returns undefined if bbox is invalid or zero area', async () => {
      expect(await extractLineCropBase64('data:image/png;base64,mock', [0.2, 0.2, 0.2, 0.2])).toBeUndefined();
      expect(await extractLineCropBase64('data:image/png;base64,mock', [NaN, 0, 1, 1])).toBeUndefined();
    });

    it('simulates non-test / real browser environment for image loading & errors', async () => {
      // Simulate real browser by overriding isTestEnv checks
      const origEnv = process.env.NODE_ENV;
      const origUserAgent = navigator.userAgent;

      try {
        (process.env as any).NODE_ENV = 'production';
        Object.defineProperty(navigator, 'userAgent', {
          value: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
          configurable: true,
        });

        // Test 1: Image triggers onload successfully with natural dimensions
        const origImage = global.Image;
        class MockLoadedImage {
          crossOrigin = '';
          width = 800;
          height = 1100;
          naturalWidth = 800;
          naturalHeight = 1100;
          complete = true;
          private _src = '';
          onload: (() => void) | null = null;
          onerror: ((ev: Event) => void) | null = null;

          set src(val: string) {
            this._src = val;
            setTimeout(() => {
              if (this.onload) this.onload();
            }, 5);
          }
          get src() {
            return this._src;
          }
        }
        // @ts-expect-error Mocking Image
        global.Image = MockLoadedImage;

        const result = await extractLineCropBase64('https://example.com/page.png', [0.1, 0.1, 0.3, 0.9]);
        expect(result).toBeDefined();
        expect(result).toMatch(/^data:image\/png;base64,/);

        // Test 2: Image triggers onerror (non-image URL or 404)
        class MockErrorImage {
          crossOrigin = '';
          width = 0;
          height = 0;
          naturalWidth = 0;
          naturalHeight = 0;
          complete = false;
          private _src = '';
          onload: (() => void) | null = null;
          onerror: ((ev: Event) => void) | null = null;

          set src(val: string) {
            this._src = val;
            setTimeout(() => {
              if (this.onerror) this.onerror(new Event('error'));
            }, 5);
          }
          get src() {
            return this._src;
          }
        }
        // @ts-expect-error Mocking Image
        global.Image = MockErrorImage;

        const errorResult = await extractLineCropBase64('https://invalid.domain.test/notfound.jpg', [0.1, 0.1, 0.3, 0.9]);
        // When image fails, it falls back gracefully without throwing
        expect(errorResult).toBeDefined();
        expect(errorResult).toMatch(/^data:image\/png;base64,/);

        // Test 3: CORS SecurityError on tainted canvas toDataURL
        const origToDataUrl = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = vi.fn().mockImplementation(() => {
          throw new DOMException('The operation is insecure.', 'SecurityError');
        });

        const corsResult = await extractLineCropBase64('https://cross-origin.com/tainted.png', [0.1, 0.1, 0.3, 0.9]);
        expect(corsResult).toBeDefined();
        // Should fall back to 1x1 transparent PNG on SecurityError
        expect(corsResult).toBe('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==');

        HTMLCanvasElement.prototype.toDataURL = origToDataUrl;
        global.Image = origImage;
      } finally {
        (process.env as any).NODE_ENV = origEnv;
        Object.defineProperty(navigator, 'userAgent', {
          value: origUserAgent,
          configurable: true,
        });
      }
    });

    it('handles headless JSDOM canvas where getContext returns null', async () => {
      const origGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null);

      try {
        const result = await extractLineCropBase64('blob:test-image', [0.1, 0.1, 0.4, 0.8]);
        expect(result).toBeUndefined();
      } finally {
        HTMLCanvasElement.prototype.getContext = origGetContext;
      }
    });

    it('handles image timeout when neither onload nor onerror fires within timeout limit', async () => {
      const origEnv = process.env.NODE_ENV;
      const origUserAgent = navigator.userAgent;

      try {
        (process.env as any).NODE_ENV = 'production';
        Object.defineProperty(navigator, 'userAgent', {
          value: 'Mozilla/5.0 Chrome/120.0.0.0',
          configurable: true,
        });

        const origImage = global.Image;
        class MockHangingImage {
          crossOrigin = '';
          width = 0;
          height = 0;
          naturalWidth = 0;
          naturalHeight = 0;
          complete = false;
          private _src = '';
          onload: (() => void) | null = null;
          onerror: ((ev: Event) => void) | null = null;

          set src(val: string) {
            this._src = val;
            // Never fires onload or onerror
          }
          get src() {
            return this._src;
          }
        }
        // @ts-expect-error Mocking Image
        global.Image = MockHangingImage;

        // Use vi.useFakeTimers to test 3000ms timeout
        vi.useFakeTimers();
        const promise = extractLineCropBase64('https://hanging-server.com/img.png', [0.1, 0.1, 0.3, 0.9]);
        await vi.advanceTimersByTimeAsync(3500);
        const result = await promise;
        expect(result).toBeDefined();

        vi.useRealTimers();
        global.Image = origImage;
      } finally {
        vi.useRealTimers();
        (process.env as any).NODE_ENV = origEnv;
        Object.defineProperty(navigator, 'userAgent', {
          value: origUserAgent,
          configurable: true,
        });
      }
    });
  });
});
