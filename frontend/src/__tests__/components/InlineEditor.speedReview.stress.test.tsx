/**
 * frontend/src/__tests__/components/InlineEditor.speedReview.stress.test.tsx
 * Dedicated Empirical Stress Test Suite for InlineEditor Speed Review Queue,
 * Rapid Shortcut Navigation, Focus Management, and RxNorm Autocomplete Popovers.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult, LineItem, WordToken } from '@/types/ocr';
import { bboxToPolygon } from '@/lib/transformUtils';

describe('InlineEditor Speed Review Queue & Keyboard Interaction Stress Suite', () => {
  const createSyntheticDocumentPage = (
    tokenCount: number,
    lowConfidenceRatio: number = 1.0
  ): PageResult => {
    const tokensPerLine = 4;
    const lineCount = Math.ceil(tokenCount / tokensPerLine);
    const lines: LineItem[] = [];

    let count = 0;
    for (let l = 0; l < lineCount; l++) {
      const words: WordToken[] = [];
      for (let w = 0; w < tokensPerLine && count < tokenCount; w++) {
        const isLowConf = count / tokenCount < lowConfidenceRatio;
        const confidence = isLowConf ? 0.45 : 0.96;
        const text = isLowConf
          ? ['Amoxcilin', 'Metformn', 'Atorvasttn', 'Lisinoprl', 'Omeprazl', 'Loratadn'][count % 6]
          : `Token_${count}`;

        words.push({
          word_id: `w_tok_${count}`,
          text,
          original_text: text,
          confidence,
          bbox: [0.1 + l * 0.05, 0.1 + w * 0.2, 0.14 + l * 0.05, 0.28 + w * 0.2],
          polygon: bboxToPolygon([0.1 + l * 0.05, 0.1 + w * 0.2, 0.14 + l * 0.05, 0.28 + w * 0.2]),
        });
        count++;
      }

      lines.push({
        line_id: `l_line_${l}`,
        line_number: l + 1,
        text: words.map((w) => w.text).join(' '),
        original_text: words.map((w) => w.text).join(' '),
        confidence: words.reduce((sum, w) => sum + (w.confidence ?? 0), 0) / (words.length || 1),
        bbox: [0.1 + l * 0.05, 0.1, 0.14 + l * 0.05, 0.9],
        polygon: bboxToPolygon([0.1 + l * 0.05, 0.1, 0.14 + l * 0.05, 0.9]),
        words,
      });
    }

    return {
      page_number: 1,
      width: 1200,
      height: 1600,
      mean_confidence: lines.reduce((sum, l) => sum + (l.confidence ?? 0), 0) / (lines.length || 1),
      full_text: lines.map((l) => l.text).join('\n'),
      lines,
    };
  };

  describe('1. Boundary Conditions: Zero and High Volume Queue', () => {
    it('handles 0 low-confidence tokens cleanly with empty state', () => {
      const cleanDoc = createSyntheticDocumentPage(10, 0.0); // all high confidence
      render(<InlineEditor enableMedicalSuggestions page={cleanDoc} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      expect(speedTab).toHaveTextContent('Speed Review (0)');

      fireEvent.click(speedTab);

      expect(screen.getByText('All Words Verified!')).toBeInTheDocument();
      const returnBtn = screen.getByTestId('btn-return-lines');
      fireEvent.click(returnBtn);

      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });

    it('populates queue for 80 low-confidence tokens and tracks index progression', () => {
      const largeDoc = createSyntheticDocumentPage(80, 1.0);
      render(<InlineEditor enableMedicalSuggestions page={largeDoc} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      expect(speedTab).toHaveTextContent('Speed Review (80)');

      fireEvent.click(speedTab);

      const panel = screen.getByTestId('speed-review-panel');
      expect(panel).toHaveTextContent(/Reviewing Uncertain Word 1 of 80/);
    });
  });

  describe('2. Keyboard Navigation & Quick Picks Stress Test', () => {
    it('supports 1-5 number quick-picks, Enter commit, Tab skipping, and Shift+Tab back navigation', async () => {
      const onWordChange = vi.fn();
      const doc = createSyntheticDocumentPage(6, 1.0);

      render(<InlineEditor enableMedicalSuggestions page={doc} onWordChange={onWordChange} />);

      // Switch to Speed Review tab
      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const input = screen.getByTestId('speed-review-input');

      // 1. Pick suggestion 1 with key '1'
      await waitFor(() => {
        expect(screen.getByTestId('speed-suggestion-1')).toBeInTheDocument();
      });
      fireEvent.keyDown(input, { key: '1', code: 'Digit1' });
      expect(onWordChange).toHaveBeenCalledWith('l_line_0', 'w_tok_0', expect.stringMatching(/Amoxicillin/i));

      // 2. We should now be at Word 2 of 6
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 2 of 6/);

      // Edit text and press Enter
      fireEvent.change(input, { target: { value: 'Metformin 1000mg' } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
      expect(onWordChange).toHaveBeenCalledWith('l_line_0', 'w_tok_1', 'Metformin 1000mg');

      // 3. We should now be at Word 3 of 6
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 3 of 6/);

      // Skip forward using Tab
      fireEvent.keyDown(input, { key: 'Tab', code: 'Tab' });
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 4 of 6/);

      // Step backward using Shift+Tab
      fireEvent.keyDown(input, { key: 'Tab', code: 'Tab', shiftKey: true });
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 3 of 6/);

      // Prev button step to 2
      const prevBtn = screen.getByTestId('btn-speed-prev');
      fireEvent.click(prevBtn);
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 2 of 6/);

      // Prev button step to 1
      fireEvent.click(prevBtn);
      expect(screen.getByTestId('speed-review-panel')).toHaveTextContent(/Reviewing Uncertain Word 1 of 6/);
      expect(prevBtn).toBeDisabled();

      // Escape key returns to structured line view
      fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' });
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });

    it('advances through entire queue to automatic completion', () => {
      const onWordChange = vi.fn();
      const doc = createSyntheticDocumentPage(5, 1.0);

      render(<InlineEditor enableMedicalSuggestions page={doc} onWordChange={onWordChange} />);

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const acceptBtn = screen.getByTestId('btn-speed-accept');

      for (let i = 0; i < 5; i++) {
        fireEvent.click(acceptBtn);
      }

      // Automatically transitions back to structured editor once queue is exhausted
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
      expect(onWordChange).toHaveBeenCalledTimes(5);
    });
  });

  describe('3. Structured Mode Autocomplete Popover', () => {
    it('opens autocomplete list on word chip selection and accepts keyboard choice', async () => {
      const onWordChange = vi.fn();
      const doc = createSyntheticDocumentPage(3, 1.0);

      render(<InlineEditor enableMedicalSuggestions page={doc} onWordChange={onWordChange} />);

      // Click word chip
      const chip = screen.getByTestId('word-chip-w_tok_0');
      fireEvent.click(chip);

      const wordInput = screen.getByTestId('word-edit-input-w_tok_0');
      expect(wordInput).toBeInTheDocument();

      // Popover should appear
      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      // Keyboard ArrowDown and Enter
      fireEvent.keyDown(wordInput, { key: 'ArrowDown', code: 'ArrowDown' });
      fireEvent.keyDown(wordInput, { key: 'Enter', code: 'Enter' });

      expect(onWordChange).toHaveBeenCalled();
    });
  });

  describe('4. Raw Text Synchronous Editing', () => {
    it('dispatches onPageUpdate when editing in raw textarea', () => {
      const onPageUpdate = vi.fn();
      const doc = createSyntheticDocumentPage(4, 0.0);

      render(<InlineEditor enableMedicalSuggestions page={doc} onPageUpdate={onPageUpdate} />);

      fireEvent.click(screen.getByTestId('tab-raw'));
      const textarea = screen.getByTestId('raw-textarea');

      fireEvent.change(textarea, { target: { value: 'Updated Prescription Full Text' } });
      expect(onPageUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          full_text: 'Updated Prescription Full Text',
        })
      );
    });
  });
});
