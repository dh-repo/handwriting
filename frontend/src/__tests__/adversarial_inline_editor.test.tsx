/**
 * frontend/src/__tests__/adversarial_inline_editor.test.tsx
 * Comprehensive Empirical Adversarial Stress Test for InlineEditor & Speed Review Queue
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult, LineItem, WordToken } from '@/types/ocr';
import { bboxToPolygon } from '@/lib/transformUtils';

describe('Adversarial InlineEditor & Speed Review Queue Stress Tests', () => {
  // Helper to create synthetic page with arbitrary number of low confidence tokens
  const createSyntheticPageWithTokens = (
    totalTokens: number,
    lowConfRatio: number = 1.0
  ): PageResult => {
    const wordsPerLine = 5;
    const totalLines = Math.ceil(totalTokens / wordsPerLine);
    const lines: LineItem[] = [];

    let tokenCounter = 0;
    for (let l = 0; l < totalLines; l++) {
      const words: WordToken[] = [];
      for (let w = 0; w < wordsPerLine && tokenCounter < totalTokens; w++) {
        const isLowConf = tokenCounter / totalTokens < lowConfRatio;
        const conf = isLowConf ? 0.35 + (tokenCounter % 30) * 0.01 : 0.95;
        const text = isLowConf
          ? ['Amoxcilin', 'Metformn', 'Atorvasttn', 'Lisinoprl', 'Omeprazl', 'Loratadn', 'Prednisn'][tokenCounter % 7]
          : 'NormalWord';

        words.push({
          word_id: `w_synth_${tokenCounter}`,
          text,
          original_text: text,
          confidence: conf,
          bbox: [0.1 + l * 0.05, 0.1 + w * 0.15, 0.14 + l * 0.05, 0.22 + w * 0.15],
          polygon: bboxToPolygon([0.1 + l * 0.05, 0.1 + w * 0.15, 0.14 + l * 0.05, 0.22 + w * 0.15]),
        });
        tokenCounter++;
      }

      lines.push({
        line_id: `l_synth_${l}`,
        line_number: l + 1,
        text: words.map((w) => w.text).join(' '),
        original_text: words.map((w) => w.text).join(' '),
        confidence: words.reduce((acc, w) => acc + (w.confidence ?? 0), 0) / (words.length || 1),
        bbox: [0.1 + l * 0.05, 0.1, 0.14 + l * 0.05, 0.9],
        polygon: bboxToPolygon([0.1 + l * 0.05, 0.1, 0.14 + l * 0.05, 0.9]),
        words,
      });
    }

    return {
      page_number: 1,
      width: 1200,
      height: 1600,
      mean_confidence: lines.reduce((acc, l) => acc + (l.confidence ?? 0), 0) / (lines.length || 1),
      full_text: lines.map((l) => l.text).join('\n'),
      lines,
    };
  };

  // =========================================================================
  // 1. BOUNDARY: ZERO (0) LOW CONFIDENCE TOKENS
  // =========================================================================
  describe('1. Zero (0) Low Confidence Tokens Boundary', () => {
    it('renders empty state "All Words Verified!" when document has 0 low-confidence tokens', () => {
      const pristinePage = createSyntheticPageWithTokens(20, 0.0); // 0% low conf
      render(<InlineEditor enableMedicalSuggestions page={pristinePage} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      expect(speedTab).toHaveTextContent('Speed Review (0)');

      fireEvent.click(speedTab);

      expect(screen.getByText('All Words Verified!')).toBeInTheDocument();
      expect(screen.getByTestId('btn-return-lines')).toBeInTheDocument();

      // Return to structured editor
      fireEvent.click(screen.getByTestId('btn-return-lines'));
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // 2. MASSIVE QUEUE: 100+ LOW CONFIDENCE TOKENS STRESS
  // =========================================================================
  describe('2. Massive 100+ Low Confidence Tokens Queue Stress', () => {
    it('initializes and manages queue with 120 low-confidence tokens', () => {
      const massivePage = createSyntheticPageWithTokens(120, 1.0);
      render(<InlineEditor enableMedicalSuggestions page={massivePage} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      expect(speedTab).toHaveTextContent('Speed Review (120)');

      fireEvent.click(speedTab);

      const panel = screen.getByTestId('speed-review-panel');
      expect(panel).toBeInTheDocument();
      expect(panel).toHaveTextContent(/Reviewing Uncertain Word 1 of 120/);
    });

    it('executes rapid keypresses (1-5 quick pick, Enter, Tab, Shift+Tab, Escape)', async () => {
      const onWordChange = vi.fn();
      const testPage = createSyntheticPageWithTokens(10, 1.0);

      render(<InlineEditor enableMedicalSuggestions page={testPage} onWordChange={onWordChange} />);

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const input = screen.getByTestId('speed-review-input');

      // 1. Quick pick key '1' for first item (Amoxcilin -> Amoxicillin)
      await waitFor(() => {
        expect(screen.getByTestId('speed-suggestion-1')).toBeInTheDocument();
      });
      fireEvent.keyDown(input, { key: '1', code: 'Digit1' });
      expect(onWordChange).toHaveBeenCalledWith('l_synth_0', 'w_synth_0', expect.stringMatching(/Amoxicillin/i));

      // 2. Now on Item 2 of 10. Type custom correction and press Enter
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 2 of 10/);
      fireEvent.change(input, { target: { value: 'Metformin HCl 500mg' } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
      expect(onWordChange).toHaveBeenCalledWith('l_synth_0', 'w_synth_1', 'Metformin HCl 500mg');

      // 3. Now on Item 3 of 10. Skip using Tab key
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 3 of 10/);
      fireEvent.keyDown(input, { key: 'Tab', code: 'Tab' });

      // 4. Now on Item 4 of 10. Step backwards using Shift+Tab
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 4 of 10/);
      fireEvent.keyDown(input, { key: 'Tab', code: 'Tab', shiftKey: true });
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 3 of 10/);

      // 5. Test Prev button at Item 3 -> Steps to Item 2
      const btnPrev = screen.getByTestId('btn-speed-prev');
      fireEvent.click(btnPrev);
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 2 of 10/);

      // 6. Test Prev button at Item 2 -> Steps to Item 1
      fireEvent.click(btnPrev);
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 1 of 10/);
      expect(btnPrev).toBeDisabled();

      // 7. Shift+Tab on first item (index 0) remains at index 0
      fireEvent.keyDown(input, { key: 'Tab', code: 'Tab', shiftKey: true });
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 1 of 10/);

      // 8. Press Escape -> Switches back to structured editor
      fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' });
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });

    it('completes rapid high-throughput triage through all 100+ items to completion', () => {
      const onWordChange = vi.fn();
      const largePage = createSyntheticPageWithTokens(100, 1.0);

      render(<InlineEditor enableMedicalSuggestions page={largePage} onWordChange={onWordChange} />);

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const input = screen.getByTestId('speed-review-input');
      const acceptBtn = screen.getByTestId('btn-speed-accept');

      // Rapidly step through all 100 tokens using Accept
      for (let i = 0; i < 100; i++) {
        fireEvent.click(acceptBtn);
      }

      // After last item, should automatically return to structured view
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
      expect(onWordChange).toHaveBeenCalledTimes(100);
    });
  });

  // =========================================================================
  // 3. STRUCTURED MODE AUTOCOMPLETE POPOVER & KEYBOARD NAVIGATION
  // =========================================================================
  describe('3. Structured Mode Autocomplete Popover & Keyboard Navigation', () => {
    it('opens autocomplete popover on word click and navigates via ArrowDown / ArrowUp / Enter', async () => {
      const onWordChange = vi.fn();
      const page = createSyntheticPageWithTokens(5, 1.0);

      render(<InlineEditor enableMedicalSuggestions page={page} onWordChange={onWordChange} />);

      // Click word chip to edit
      const wordChip = screen.getByTestId('word-chip-w_synth_0');
      fireEvent.click(wordChip);

      const wordInput = screen.getByTestId('word-edit-input-w_synth_0');
      expect(wordInput).toBeInTheDocument();

      // Autocomplete popover appears
      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      // Navigate down and up
      fireEvent.keyDown(wordInput, { key: 'ArrowDown', code: 'ArrowDown' });
      fireEvent.keyDown(wordInput, { key: 'ArrowUp', code: 'ArrowUp' });

      // Apply selection with Enter
      fireEvent.keyDown(wordInput, { key: 'Enter', code: 'Enter' });
      expect(onWordChange).toHaveBeenCalledWith('l_synth_0', 'w_synth_0', expect.any(String));
    });

    it('dismisses autocomplete popover on Escape key without submitting change', async () => {
      const page = createSyntheticPageWithTokens(5, 1.0);
      render(<InlineEditor enableMedicalSuggestions page={page} />);

      const wordChip = screen.getByTestId('word-chip-w_synth_0');
      fireEvent.click(wordChip);

      const wordInput = screen.getByTestId('word-edit-input-w_synth_0');
      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      fireEvent.keyDown(wordInput, { key: 'Escape', code: 'Escape' });
      expect(screen.queryByTestId('medical-autocomplete-popover')).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // 4. PLAIN TEXT MODE SYNC
  // =========================================================================
  describe('4. Plain Text Mode Synchronous Editing', () => {
    it('updates full document text in raw plain text tab', () => {
      const onPageUpdate = vi.fn();
      const page = createSyntheticPageWithTokens(6, 0.0);

      render(<InlineEditor enableMedicalSuggestions page={page} onPageUpdate={onPageUpdate} />);

      const rawTab = screen.getByTestId('tab-raw');
      fireEvent.click(rawTab);

      const textarea = screen.getByTestId('raw-textarea');
      expect(textarea).toBeInTheDocument();

      fireEvent.change(textarea, { target: { value: 'Direct Edited Plain Text' } });
      expect(onPageUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          full_text: 'Direct Edited Plain Text',
        })
      );
    });
  });
});
