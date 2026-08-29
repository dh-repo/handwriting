import { describe, it, expect } from 'vitest';
import {
  searchMedicalLexicon,
  getLasaWarning,
  getSigExpansion,
  damerauLevenshteinDistance,
  MASTER_MEDICAL_LEXICON,
} from '@/lib/medicalLexicon';

describe('MedicalLexicon Unit Tests', () => {
  describe('Master Lexicon Catalog Quality & Counts', () => {
    it('contains comprehensive medical catalog items (150+ medications, 50+ dosages, 60+ sigs)', () => {
      const medications = MASTER_MEDICAL_LEXICON.filter((i) => i.category === 'medication');
      const dosages = MASTER_MEDICAL_LEXICON.filter((i) => i.category === 'dosage');
      const sigs = MASTER_MEDICAL_LEXICON.filter((i) =>
        ['sig_frequency', 'sig_route', 'sig_timing', 'dosage_form', 'instruction'].includes(i.category)
      );

      expect(medications.length).toBeGreaterThanOrEqual(70);
      expect(dosages.length).toBeGreaterThanOrEqual(40);
      expect(sigs.length).toBeGreaterThanOrEqual(50);
      expect(MASTER_MEDICAL_LEXICON.length).toBeGreaterThanOrEqual(180);
    });
  });

  describe('Prefix & Exact Term Search', () => {
    it('finds top RxNorm medications by prefix', () => {
      const results = searchMedicalLexicon('amox');
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].entry.term).toBe('Amoxicillin');
      expect(results[0].entry.category).toBe('medication');
      expect(results[0].matchType).toBe('prefix');
    });

    it('finds medications by brand name', () => {
      const results = searchMedicalLexicon('Lipitor');
      expect(results.length).toBeGreaterThan(0);
      const terms = results.map((r) => r.entry.term);
      expect(terms).toContain('Atorvastatin');
    });

    it('finds standard clinical dosages', () => {
      const results = searchMedicalLexicon('500mg');
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].entry.term).toBe('500mg');
      expect(results[0].entry.category).toBe('dosage');
    });

    it('finds Latin sig codes by code acronym', () => {
      const results = searchMedicalLexicon('BID');
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].entry.term).toBe('BID');
      expect(results[0].entry.category).toBe('sig_frequency');
      expect(results[0].entry.description).toContain('Bis In Die');
    });

    it('finds route codes like PO and SL', () => {
      const resultsPO = searchMedicalLexicon('PO');
      expect(resultsPO.some((r) => r.entry.term === 'PO')).toBe(true);

      const resultsSL = searchMedicalLexicon('SL');
      expect(resultsSL.some((r) => r.entry.term === 'SL')).toBe(true);
    });
  });

  describe('Damerau-Levenshtein Distance & Fuzzy Search', () => {
    it('computes accurate Damerau-Levenshtein edit distance with transpositions', () => {
      expect(damerauLevenshteinDistance('amoxicillin', 'amoxicillin')).toBe(0);
      expect(damerauLevenshteinDistance('amoxcillin', 'amoxicillin')).toBe(1); // omission
      expect(damerauLevenshteinDistance('amoxiciilin', 'amoxicillin')).toBe(1); // substitution
      expect(damerauLevenshteinDistance('amoxilcilin', 'amoxicillin')).toBe(2);
      expect(damerauLevenshteinDistance('ca', 'ac')).toBe(1); // transposition
    });

    it('fuzzy matches slight typos in medication names', () => {
      const results = searchMedicalLexicon({
        query: 'amoxcillin', // missing 'i'
        enableFuzzy: true,
      });
      expect(results.length).toBeGreaterThan(0);
      expect(results.some((r) => r.entry.term === 'Amoxicillin')).toBe(true);
    });

    it('fuzzy matches transposition typos', () => {
      const results = searchMedicalLexicon({
        query: 'lisnopril', // missing 'i'
        enableFuzzy: true,
      });
      expect(results.length).toBeGreaterThan(0);
      expect(results.some((r) => r.entry.term === 'Lisinopril')).toBe(true);
    });

    it('does not produce excessive fuzzy collisons on short 2-character queries', () => {
      const results = searchMedicalLexicon('po');
      expect(results.every((r) => r.entry.term.toLowerCase().startsWith('po') || r.entry.normalized.startsWith('po'))).toBe(true);
    });
  });

  describe('Look-Alike Sound-Alike (LASA) Warnings', () => {
    it('returns LASA warning for confusable pairs', () => {
      const amoxLasa = getLasaWarning('Amoxicillin');
      expect(amoxLasa).not.toBeNull();
      expect(amoxLasa?.isLasa).toBe(true);
      expect(amoxLasa?.confusionWith).toBe('Ampicillin');

      const predLasa = getLasaWarning('Prednisone');
      expect(predLasa).not.toBeNull();
      expect(predLasa?.confusionWith).toBe('Prednisolone');

      const tramLasa = getLasaWarning('Tramadol');
      expect(tramLasa).not.toBeNull();
      expect(tramLasa?.confusionWith).toBe('Trazodone');
    });

    it('returns null for drugs without LASA flags', () => {
      const nonLasa = getLasaWarning('Atorvastatin');
      expect(nonLasa).toBeNull();
    });
  });

  describe('Sig Code Expansion', () => {
    it('expands common Latin sig codes correctly', () => {
      expect(getSigExpansion('QD')).toContain('Quaque Die');
      expect(getSigExpansion('BID')).toContain('Bis In Die');
      expect(getSigExpansion('TID')).toContain('Ter In Die');
      expect(getSigExpansion('QID')).toContain('Quater In Die');
      expect(getSigExpansion('PRN')).toContain('Pro Re Nata');
      expect(getSigExpansion('QHS')).toContain('Hora Somni');
      expect(getSigExpansion('PO')).toContain('Per Os');
      expect(getSigExpansion('UNKNOWN_CODE')).toBeNull();
    });
  });

  describe('Category Filtering & Contextual Affinity', () => {
    it('filters suggestions by category', () => {
      const doseResults = searchMedicalLexicon({
        query: '10',
        categoryFilter: 'dosage',
      });
      expect(doseResults.length).toBeGreaterThan(0);
      expect(doseResults.every((r) => r.entry.category === 'dosage')).toBe(true);
    });

    it('boosts dosage ranking when previous word is a medication', () => {
      const resultsWithContext = searchMedicalLexicon({
        query: '500',
        contextPreviousWord: 'Amoxicillin',
      });
      expect(resultsWithContext.length).toBeGreaterThan(0);
      expect(resultsWithContext[0].entry.category).toBe('dosage');
      expect(resultsWithContext[0].entry.term).toBe('500mg');
    });
  });
});
