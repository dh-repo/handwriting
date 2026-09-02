import { describe, expect, it } from 'vitest';
import {
  decisionFor,
  findSignatureCandidates,
  isGeneralPurposeSignatureLine,
  pendingReviewCount,
  upsertSignatureReview,
} from '../../lib/signatureCandidates';
import { SAMPLE_LEGAL_CONTRACT } from '../../lib/sampleDocuments';
import { LineItem, PageResult } from '../../types/ocr';

function pageWithLines(texts: string[]): PageResult {
  const lines: LineItem[] = texts.map((text, idx) => ({
    line_id: `p1_l${idx + 1}`,
    text,
    confidence: 0.9,
    bbox: [0.1 * idx, 0.1, 0.1 * idx + 0.08, 0.9],
    words: [],
  }));
  return {
    page_number: 1,
    width: 800,
    height: 600,
    full_text: texts.join('\n'),
    mean_confidence: 0.9,
    lines,
  };
}

describe('general-purpose signature candidates', () => {
  it('finds sign-off language on a legal contract sample', () => {
    const found = findSignatureCandidates(SAMPLE_LEGAL_CONTRACT.pages[0]);
    expect(found.length).toBeGreaterThan(0);
    expect(found.some((c) => /signature/i.test(c.text))).toBe(true);
    expect(found.every((c) => c.reviewRequired)).toBe(true);
  });

  it('does not treat medical credentials as a signature', () => {
    expect(isGeneralPurposeSignatureLine('Dr. Ulysses Eisenhower, MD')).toBe(false);
    expect(isGeneralPurposeSignatureLine('Prescriber NPI 1234567893')).toBe(false);
    expect(isGeneralPurposeSignatureLine('DEA AB1234567')).toBe(false);
    const found = findSignatureCandidates(
      pageWithLines(['Amoxicillin 500mg PO TID', 'Dr. Jane Roe, MD', 'NPI 1234567893'])
    );
    expect(found).toHaveLength(0);
  });

  it('does not treat the last line of a page as a signature by position', () => {
    const found = findSignatureCandidates(
      pageWithLines(['Please file this letter with the rest of the packet.', 'Sincerely'])
    );
    expect(found).toHaveLength(0);
  });

  it('accepts general-purpose signed / witness language', () => {
    expect(isGeneralPurposeSignatureLine('Signed by Arthur Pendelton')).toBe(true);
    expect(isGeneralPurposeSignatureLine('IN WITNESS WHEREOF, the parties have executed this Agreement.')).toBe(true);
    expect(isGeneralPurposeSignatureLine('Authorized Signature: A. Pendelton')).toBe(true);
  });

  it('records a human decision without treating it as a match score', () => {
    const page = SAMPLE_LEGAL_CONTRACT.pages[0];
    const candidates = findSignatureCandidates(page);
    const lineId = candidates[0].lineId;
    const reviews = upsertSignatureReview([], {
      page_number: page.page_number,
      line_id: lineId,
      kind: candidates[0].kind,
      decision: 'accepted',
      decided_at: '2026-08-29T00:00:00Z',
    });
    expect(decisionFor(reviews, page.page_number, lineId)).toBe('accepted');
    expect(pendingReviewCount(candidates, reviews, page.page_number)).toBe(candidates.length - 1);
  });
});
