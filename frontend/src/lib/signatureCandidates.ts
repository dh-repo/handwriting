/**
 * General-purpose signature candidate detection.
 *
 * Finds sign-off language on letters, contracts, and forms.
 * Does not treat medical credentials (Dr., MD, DEA, NPI) as signatures.
 * Does not claim a match or a verified hand.
 */

import { LineItem, PageResult, SignatureDecision, SignatureReviewRecord } from '../types/ocr';

export type SignatureKind = 'sign_off_line' | 'witness_mark' | 'handwritten_mark';

export interface SignatureCandidate {
  lineId: string;
  text: string;
  confidence: number;
  kind: SignatureKind;
  reviewRequired: true;
}

const SIGN_OFF = /\b(signature|signed|sign-off|signoff|undersigned)\b/i;
const WITNESS = /\b(witness|notary|authorized\s+sign)/i;
const MEDICAL_CREDENTIAL =
  /\b(dr\.?|m\.?d\.?|d\.?o\.?|mbbs|pa-c|npi|dea|rx|prescription|prescriber)\b/i;

function classify(text: string): SignatureKind {
  if (WITNESS.test(text)) {
    return 'witness_mark';
  }
  if (SIGN_OFF.test(text)) {
    return 'sign_off_line';
  }
  return 'handwritten_mark';
}

export function isGeneralPurposeSignatureLine(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length < 4) {
    return false;
  }
  if (MEDICAL_CREDENTIAL.test(trimmed) && !SIGN_OFF.test(trimmed) && !WITNESS.test(trimmed)) {
    return false;
  }
  return SIGN_OFF.test(trimmed) || WITNESS.test(trimmed);
}

export function findSignatureCandidates(page: PageResult): SignatureCandidate[] {
  const candidates: SignatureCandidate[] = [];
  page.lines.forEach((line: LineItem) => {
    if (!isGeneralPurposeSignatureLine(line.text)) {
      return;
    }
    candidates.push({
      lineId: line.line_id,
      text: line.text,
      confidence: line.confidence ?? 0,
      kind: classify(line.text),
      reviewRequired: true,
    });
  });
  return candidates;
}

export function kindLabel(kind: SignatureKind): string {
  switch (kind) {
    case 'sign_off_line':
      return 'Sign-off line';
    case 'witness_mark':
      return 'Witness / authorized mark';
    case 'handwritten_mark':
      return 'Handwritten mark';
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

export function decisionLabel(decision: SignatureDecision): string {
  switch (decision) {
    case 'pending':
      return 'Pending review';
    case 'accepted':
      return 'Accepted as present';
    case 'rejected':
      return 'Rejected as not a mark';
    default: {
      const _exhaustive: never = decision;
      return _exhaustive;
    }
  }
}

export function reviewKey(pageNumber: number, lineId: string): string {
  return `${pageNumber}:${lineId}`;
}

export function decisionFor(
  reviews: SignatureReviewRecord[] | undefined,
  pageNumber: number,
  lineId: string
): SignatureDecision {
  const found = (reviews ?? []).find(
    (row) => row.page_number === pageNumber && row.line_id === lineId
  );
  return found?.decision ?? 'pending';
}

export function upsertSignatureReview(
  reviews: SignatureReviewRecord[],
  next: SignatureReviewRecord
): SignatureReviewRecord[] {
  const without = reviews.filter(
    (row) => !(row.page_number === next.page_number && row.line_id === next.line_id)
  );
  return [...without, next];
}

export function pendingReviewCount(
  candidates: SignatureCandidate[],
  reviews: SignatureReviewRecord[] | undefined,
  pageNumber: number
): number {
  return candidates.filter(
    (candidate) => decisionFor(reviews, pageNumber, candidate.lineId) === 'pending'
  ).length;
}
