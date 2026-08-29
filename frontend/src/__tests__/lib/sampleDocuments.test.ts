import { describe, it, expect } from 'vitest';
import {
  SAMPLE_CLEAN_CURSIVE,
  SAMPLE_MESSY_CURSIVE,
  SAMPLE_PRESCRIPTION,
  SAMPLE_MULTIPAGE,
  SAMPLE_PRESETS,
} from '@/lib/sampleDocuments';

describe('sampleDocuments', () => {
  it('contains 4 valid sample presets in SAMPLE_PRESETS registry', () => {
    expect(Object.keys(SAMPLE_PRESETS)).toEqual([
      'sample_clean_cursive',
      'sample_messy_cursive',
      'sample_prescription',
      'sample_multipage',
    ]);
  });

  it('validates clean cursive preset coordinates and tokens', () => {
    expect(SAMPLE_CLEAN_CURSIVE.pages[0].lines).toHaveLength(4);
    const firstLine = SAMPLE_CLEAN_CURSIVE.pages[0].lines[0];
    expect(firstLine.text).toBe('The quick brown fox jumps over the lazy dog.');
    expect(firstLine.words).toHaveLength(9);
    expect(firstLine.bbox[0]).toBeCloseTo(0.23667, 4);
    expect(firstLine.bbox[1]).toBeCloseTo(0.17875, 4);
  });

  it('validates messy cursive preset contains low-confidence tokens for speed review', () => {
    const hasLowConfWord = SAMPLE_MESSY_CURSIVE.pages[0].lines.some((line) =>
      line.words.some((word) => word.confidence < 0.70)
    );
    expect(hasLowConfWord).toBe(true);
  });

  it('validates multi-page document pagination', () => {
    expect(SAMPLE_MULTIPAGE.total_pages).toBe(3);
    expect(SAMPLE_MULTIPAGE.pages).toHaveLength(3);
    expect(SAMPLE_MULTIPAGE.pages[0].page_number).toBe(1);
    expect(SAMPLE_MULTIPAGE.pages[1].page_number).toBe(2);
    expect(SAMPLE_MULTIPAGE.pages[2].page_number).toBe(3);
  });
});
