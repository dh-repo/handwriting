/**
 * frontend/src/__tests__/lib/medicalLexicon.stress.test.ts
 * Comprehensive Stress & Empirical Challenge Test Suite for Medical Lexicon
 */

import { describe, it, expect } from 'vitest';
import {
  searchMedicalLexicon,
  getLasaWarning,
  getSigExpansion,
  damerauLevenshteinDistance,
  MASTER_MEDICAL_LEXICON,
} from '@/lib/medicalLexicon';

describe('MedicalLexicon Stress & Adversarial Challenge Tests', () => {
  // =========================================================================
  // 1. EXTREME INPUTS (EMPTY, WHITESPACE, NULL-LIKE, SPECIAL CHARS, UNICODE, LONG STRINGS)
  // =========================================================================
  describe('1. Extreme & Malformed Input Handling', () => {
    it('returns empty array for empty, whitespace, or degenerate query strings', () => {
      expect(searchMedicalLexicon('')).toEqual([]);
      expect(searchMedicalLexicon('   ')).toEqual([]);
      expect(searchMedicalLexicon('\t\n\r  ')).toEqual([]);
      expect(searchMedicalLexicon({ query: '' })).toEqual([]);
      expect(searchMedicalLexicon({ query: '   ' })).toEqual([]);
      expect(searchMedicalLexicon(undefined as any)).toEqual([]);
      expect(searchMedicalLexicon(null as any)).toEqual([]);
    });

    it('handles query strings composed exclusively of punctuation and symbols without crashing', () => {
      const symbols = [
        '!@#$%^&*()_+=~`',
        '[]{}|;:\'",.<>?/\\',
        '---+++===***',
        '/*<![CDATA[*/',
        '<script>alert("xss")</script>',
        'DROP TABLE medications;--',
      ];

      for (const sym of symbols) {
        const results = searchMedicalLexicon(sym);
        expect(Array.isArray(results)).toBe(true);
      }
    });

    it('handles multi-lingual unicode, emojis, and non-ASCII characters gracefully', () => {
      const unicodeQueries = [
        '💊💉🩺', // Medical emojis
        'Амоксициллин', // Russian Cyrillic for Amoxicillin
        'أموكسيسيلين', // Arabic
        '阿莫西林', // Simplified Chinese
        'Café au lait', // French accents
        'Amoxi\u200Bcillin', // Zero-width space embedded
        '\uFEFFAmoxicillin', // Byte Order Mark
        '𝔄𝔪𝔬𝔵𝔦𝔠𝔦𝔩𝔩𝔦𝔫', // Fraktur Unicode
      ];

      for (const uq of unicodeQueries) {
        const res = searchMedicalLexicon(uq);
        expect(Array.isArray(res)).toBe(true);
      }
    });

    it('handles massive repetitive string queries without timeout or stack overflow', () => {
      const longQuery1 = 'a'.repeat(5000);
      const longQuery2 = 'Amoxicillin500mgPOTID'.repeat(200);
      const randomJunk = Array.from({ length: 2000 }, (_, i) => String.fromCharCode(97 + (i % 26))).join('');

      const t0 = performance.now();
      const res1 = searchMedicalLexicon(longQuery1);
      const res2 = searchMedicalLexicon(longQuery2);
      const res3 = searchMedicalLexicon(randomJunk);
      const elapsed = performance.now() - t0;

      expect(Array.isArray(res1)).toBe(true);
      expect(Array.isArray(res2)).toBe(true);
      expect(Array.isArray(res3)).toBe(true);
      expect(elapsed).toBeLessThan(1000); // Must execute in < 1 second
    });
  });

  // =========================================================================
  // 2. DAMERAU-LEVENSHTEIN DISTANCE & TRANSPOSITIONS
  // =========================================================================
  describe('2. Damerau-Levenshtein Edit Distance Stress & Edge Cases', () => {
    it('accurately computes edge cases in edit distance', () => {
      // Empty strings
      expect(damerauLevenshteinDistance('', '')).toBe(0);
      expect(damerauLevenshteinDistance('abc', '')).toBe(3);
      expect(damerauLevenshteinDistance('', 'xyz')).toBe(3);

      // Identity & Case-insensitivity
      expect(damerauLevenshteinDistance('Amoxicillin', 'amoxicillin')).toBe(0);
      expect(damerauLevenshteinDistance('LISINOPRIL', 'lisinopril')).toBe(0);

      // Adjacent transpositions (Damerau property)
      expect(damerauLevenshteinDistance('ab', 'ba')).toBe(1);
      expect(damerauLevenshteinDistance('amoxciillin', 'amoxicillin')).toBe(1); // 'ci' <-> 'ic'
      expect(damerauLevenshteinDistance('prednsione', 'prednisone')).toBe(1); // 'si' <-> 'is'

      // Multiple edits
      expect(damerauLevenshteinDistance('metfrmin', 'metformin')).toBe(1); // insertion
      expect(damerauLevenshteinDistance('metforminn', 'metformin')).toBe(1); // deletion
      expect(damerauLevenshteinDistance('netformin', 'metformin')).toBe(1); // substitution
    });

    it('fuzzy search resolves complex typos within distance tolerance', () => {
      // 1-edit distance typos on long drug names
      const typos = [
        { query: 'amoxcillin', expected: 'Amoxicillin' },
        { query: 'atorvasttin', expected: 'Atorvastatin' },
        { query: 'levothyroxin', expected: 'Levothyroxine' },
        { query: 'metformn', expected: 'Metformin' },
        { query: 'gabapentn', expected: 'Gabapentin' },
        { query: 'furosmide', expected: 'Furosemide' },
        { query: 'ciprofloxcin', expected: 'Ciprofloxacin' },
      ];

      for (const { query, expected } of typos) {
        const results = searchMedicalLexicon({ query, enableFuzzy: true });
        expect(results.length).toBeGreaterThan(0);
        expect(results.some((r) => r.entry.term === expected)).toBe(true);
      }
    });
  });

  // =========================================================================
  // 3. EXACT VS PREFIX VS SUBSTRING VS FUZZY SCORING HIERARCHY
  // =========================================================================
  describe('3. Ranking & Scoring Invariants', () => {
    it('ranks exact match strictly higher than prefix, substring, and fuzzy matches', () => {
      // "Metformin" exact vs prefix vs substring
      const exactResults = searchMedicalLexicon('Metformin');
      expect(exactResults[0].entry.term).toBe('Metformin');
      expect(exactResults[0].matchType).toBe('exact');
      expect(exactResults[0].score).toBeGreaterThan(1000);

      // Prefix match score check
      const prefixResults = searchMedicalLexicon('Metform');
      expect(prefixResults[0].matchType).toBe('prefix');
      expect(prefixResults[0].score).toBeLessThan(exactResults[0].score);

      // Substring match score check ("formin" -> Metformin)
      const subResults = searchMedicalLexicon('formin');
      const metMatch = subResults.find((r) => r.entry.term === 'Metformin');
      expect(metMatch).toBeDefined();
      expect(metMatch!.matchType).toBe('substring');
      expect(metMatch!.score).toBeLessThan(prefixResults[0].score);

      // Fuzzy match score check ("metformn" -> Metformin)
      const fuzzyResults = searchMedicalLexicon('metformn');
      const fuzzyMatch = fuzzyResults.find((r) => r.entry.term === 'Metformin');
      expect(fuzzyMatch).toBeDefined();
      expect(fuzzyMatch!.matchType).toBe('fuzzy');
      expect(fuzzyMatch!.score).toBeLessThan(metMatch!.score);
    });

    it('respects maxResults parameter limits', () => {
      for (const limit of [1, 2, 5, 10, 20]) {
        const res = searchMedicalLexicon({ query: 'a', maxResults: limit });
        expect(res.length).toBeLessThanOrEqual(limit);
      }
    });

    it('correctly filters by single and multiple categories', () => {
      // Only medications
      const medOnly = searchMedicalLexicon({ query: 'a', categoryFilter: 'medication' });
      expect(medOnly.every((r) => r.entry.category === 'medication')).toBe(true);

      // Only dosages
      const doseOnly = searchMedicalLexicon({ query: '50', categoryFilter: 'dosage' });
      expect(doseOnly.every((r) => r.entry.category === 'dosage')).toBe(true);

      // Multiple categories: sig_frequency and sig_route
      const multiFilter = searchMedicalLexicon({
        query: 'q',
        categoryFilter: ['sig_frequency', 'sig_route'],
      });
      expect(multiFilter.every((r) => ['sig_frequency', 'sig_route'].includes(r.entry.category))).toBe(true);
    });

    it('boosts dosage and sig suggestions when contextPreviousWord is provided', () => {
      // Without context
      const noContext = searchMedicalLexicon({ query: '500' });
      // With medication context
      const withContext = searchMedicalLexicon({
        query: '500',
        contextPreviousWord: 'Amoxicillin',
      });

      const doseNoContext = noContext.find((r) => r.entry.term === '500mg');
      const doseWithContext = withContext.find((r) => r.entry.term === '500mg');

      expect(doseWithContext).toBeDefined();
      if (doseNoContext) {
        expect(doseWithContext!.score).toBeGreaterThan(doseNoContext.score);
      }
    });
  });

  // =========================================================================
  // 4. LOOK-ALIKE SOUND-ALIKE (LASA) SAFETY CATALOG VERIFICATION
  // =========================================================================
  describe('4. Look-Alike Sound-Alike (LASA) Safety Catalog Invariants', () => {
    it('correctly identifies all high-risk clinical LASA pairs', () => {
      const lasaPairs = [
        { term: 'Amoxicillin', confusesWith: 'Ampicillin' },
        { term: 'Ampicillin', confusesWith: 'Amoxicillin' },
        { term: 'Prednisone', confusesWith: 'Prednisolone' },
        { term: 'Prednisolone', confusesWith: 'Prednisone' },
        { term: 'Tramadol', confusesWith: 'Trazodone' },
        { term: 'Trazodone', confusesWith: 'Tramadol' },
        { term: 'Clonazepam', confusesWith: 'Clonidine' },
        { term: 'Clonidine', confusesWith: 'Clonazepam' },
        { term: 'Hydrochlorothiazide', confusesWith: 'Hydralazine' },
        { term: 'Hydralazine', confusesWith: 'Hydrochlorothiazide' },
        { term: 'Celecoxib', confusesWith: 'Celexa' },
      ];

      for (const { term, confusesWith } of lasaPairs) {
        const warning = getLasaWarning(term);
        expect(warning).not.toBeNull();
        expect(warning?.isLasa).toBe(true);
        expect(warning?.confusionWith).toBe(confusesWith);
        expect(warning?.warning).toBeDefined();
        expect(warning?.warning!.length).toBeGreaterThan(5);
      }
    });

    it('handles case-insensitive and punctuated LASA queries', () => {
      expect(getLasaWarning('amoxicillin')?.isLasa).toBe(true);
      expect(getLasaWarning('AMOXICILLIN')?.isLasa).toBe(true);
      expect(getLasaWarning('Amox-icillin')?.isLasa).toBe(true);
      expect(getLasaWarning('Lipitor')).toBeNull(); // Non-LASA
      expect(getLasaWarning('')).toBeNull();
      expect(getLasaWarning('nonexistent_drug_xyz')).toBeNull();
    });
  });

  // =========================================================================
  // 5. LATIN SIG CODE EXPANSION FIDELITY
  // =========================================================================
  describe('5. Latin Sig Code Expansions Completeness & Case-Insensitivity', () => {
    it('expands all standard Latin prescription sig frequencies', () => {
      const sigMap: Record<string, string> = {
        QD: 'Once every day',
        BID: 'Twice a day',
        TID: 'Three times a day',
        QID: 'Four times a day',
        Q4H: 'Every 4 hours',
        Q6H: 'Every 6 hours',
        Q8H: 'Every 8 hours',
        Q12H: 'Every 12 hours',
        QHS: 'Every night at bedtime',
        STAT: 'Immediately',
      };

      for (const [code, substr] of Object.entries(sigMap)) {
        const expansion = getSigExpansion(code);
        expect(expansion).not.toBeNull();
        expect(expansion).toContain(substr);
      }
    });

    it('expands all standard routes and dosage forms', () => {
      expect(getSigExpansion('PO')).toContain('By mouth');
      expect(getSigExpansion('SL')).toContain('Sublingually');
      expect(getSigExpansion('PR')).toContain('Rectally');
      expect(getSigExpansion('IM')).toContain('Intramuscular');
      expect(getSigExpansion('IV')).toContain('Intravenous');
      expect(getSigExpansion('PRN')).toContain('As needed');
      expect(getSigExpansion('TAB')).toBe('Tablet');
      expect(getSigExpansion('CAP')).toBe('Capsule');
      expect(getSigExpansion('GTT')).toContain('Drops');
    });

    it('handles lowercase, leading/trailing whitespace, and unknown sig codes', () => {
      expect(getSigExpansion('  bid  ')).toContain('Twice a day');
      expect(getSigExpansion('tid')).toContain('Three times a day');
      expect(getSigExpansion('po')).toContain('By mouth');
      expect(getSigExpansion('unknown_random_sig')).toBeNull();
      expect(getSigExpansion('')).toBeNull();
    });
  });

  // =========================================================================
  // 6. PERFORMANCE & THROUGHPUT BENCHMARK
  // =========================================================================
  describe('6. Search Throughput & Memory Stress', () => {
    it('executes 1,000 queries in < 500ms without memory degradation', () => {
      const sampleQueries = ['amox', '500', 'bid', 'lipi', 'pred', 'traz', 'met', 'po', 'tab', '10mg'];
      const t0 = performance.now();

      for (let i = 0; i < 1000; i++) {
        const q = sampleQueries[i % sampleQueries.length];
        const res = searchMedicalLexicon(q);
        expect(res.length).toBeGreaterThan(0);
      }

      const elapsedMs = performance.now() - t0;
      expect(elapsedMs).toBeLessThan(500); // Under 500ms for 1000 queries (< 0.5ms per query)
    });
  });
});
