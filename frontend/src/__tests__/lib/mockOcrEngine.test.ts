import { describe, it, expect } from 'vitest';
import { runMockOcr } from '@/lib/mockOcrEngine';

describe('mockOcrEngine', () => {
  it('returns clean cursive preset when sampleId is sample_clean_cursive', async () => {
    const result = await runMockOcr({ sampleId: 'sample_clean_cursive' });
    expect(result.document_id).toBe('sample_clean_cursive');
    expect(result.pages[0].lines).toHaveLength(4);
    expect(result.mean_confidence).toBeGreaterThan(0.90);
  });

  it('returns prescription preset when filename contains prescription', async () => {
    const result = await runMockOcr({ filename: 'doctor_prescription_rx.png' });
    expect(result.document_id).toBe('sample_prescription');
    expect(result.pages[0].lines).toHaveLength(6);
    expect(result.pages[0].full_text).toContain('Amoxicillin');
  });

  it('synthesizes realistic structured layout for arbitrary uploaded image', async () => {
    const result = await runMockOcr({ filename: 'arbitrary_handwriting_scan.jpg' });
    expect(result.total_pages).toBe(1);
    expect(result.pages[0].lines.length).toBeGreaterThanOrEqual(4);

    result.pages[0].lines.forEach((line) => {
      expect(line.bbox[0]).toBeGreaterThanOrEqual(0);
      expect(line.bbox[1]).toBeGreaterThanOrEqual(0);
      expect(line.bbox[2]).toBeLessThanOrEqual(1);
      expect(line.bbox[3]).toBeLessThanOrEqual(1);
      expect(line.words.length).toBeGreaterThan(0);
    });
  });

  it('returns multi-page document for arbitrary PDF upload', async () => {
    const result = await runMockOcr({ filename: 'clinical_intake_multi.pdf' });
    expect(result.total_pages).toBe(3);
    expect(result.pages).toHaveLength(3);
    expect(result.pages[0].lines.length).toBeGreaterThan(0);
    expect(result.pages[1].lines.length).toBeGreaterThan(0);
    expect(result.pages[2].lines.length).toBeGreaterThan(0);
  });
});
