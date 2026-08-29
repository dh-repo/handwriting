import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult } from '@/types/ocr';

describe('InlineEditor Medical Autocomplete & Speed Review Tests', () => {
  const sampleMedicalPage: PageResult = {
    page_number: 1,
    width: 800,
    height: 1000,
    full_text: 'Amox 500 po tid',
    mean_confidence: 0.75,
    lines: [
      {
        line_id: 'p1_l1',
        text: 'Amox 500 po tid',
        confidence: 0.75,
        bbox: [0.1, 0.1, 0.2, 0.9],
        words: [
          {
            word_id: 'p1_l1_w1',
            text: 'Amox',
            original_text: 'Amox',
            confidence: 0.65, // <0.70 triggers speed review
            bbox: [0.1, 0.1, 0.2, 0.3],
          },
          {
            word_id: 'p1_l1_w2',
            text: '500',
            original_text: '500',
            confidence: 0.68, // <0.70 triggers speed review
            bbox: [0.1, 0.35, 0.2, 0.5],
          },
          {
            word_id: 'p1_l1_w3',
            text: 'po',
            original_text: 'po',
            confidence: 0.95,
            bbox: [0.1, 0.55, 0.2, 0.7],
          },
          {
            word_id: 'p1_l1_w4',
            text: 'tid',
            original_text: 'tid',
            confidence: 0.92,
            bbox: [0.1, 0.75, 0.2, 0.9],
          },
        ],
      },
    ],
  };

  describe('Medical Autocomplete Popover in Structured Mode', () => {
    it('opens autocomplete popover when editing a word chip', async () => {
      const onWordChange = vi.fn();
      render(<InlineEditor page={sampleMedicalPage} onWordChange={onWordChange} />);

      const amoxWordChip = screen.getByTestId('word-chip-p1_l1_w1');
      fireEvent.click(amoxWordChip);

      const wordInput = screen.getByTestId('word-edit-input-p1_l1_w1');
      expect(wordInput).toBeInTheDocument();
      expect(wordInput).toHaveValue('Amox');

      // Popover should render with suggestions
      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      const firstItem = screen.getByTestId('autocomplete-item-0');
      expect(firstItem).toHaveTextContent('Amoxicillin');
    });

    it('navigates autocomplete popover with ArrowDown and selects with Enter', async () => {
      const onWordChange = vi.fn();
      render(<InlineEditor page={sampleMedicalPage} onWordChange={onWordChange} />);

      const amoxWordChip = screen.getByTestId('word-chip-p1_l1_w1');
      fireEvent.click(amoxWordChip);

      const wordInput = screen.getByTestId('word-edit-input-p1_l1_w1');

      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      // Press ArrowDown to navigate
      fireEvent.keyDown(wordInput, { key: 'ArrowDown' });

      // Press Enter to apply
      fireEvent.keyDown(wordInput, { key: 'Enter' });

      expect(onWordChange).toHaveBeenCalled();
    });

    it('closes popover on Escape', async () => {
      render(<InlineEditor page={sampleMedicalPage} />);

      const amoxWordChip = screen.getByTestId('word-chip-p1_l1_w1');
      fireEvent.click(amoxWordChip);

      const wordInput = screen.getByTestId('word-edit-input-p1_l1_w1');

      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      fireEvent.keyDown(wordInput, { key: 'Escape' });

      expect(screen.queryByTestId('medical-autocomplete-popover')).not.toBeInTheDocument();
    });

    it('applies suggestion when clicking directly on popover item', async () => {
      const onWordChange = vi.fn();
      render(<InlineEditor page={sampleMedicalPage} onWordChange={onWordChange} />);

      const amoxWordChip = screen.getByTestId('word-chip-p1_l1_w1');
      fireEvent.click(amoxWordChip);

      await waitFor(() => {
        expect(screen.getByTestId('medical-autocomplete-popover')).toBeInTheDocument();
      });

      const firstItem = screen.getByTestId('autocomplete-item-0');
      fireEvent.mouseDown(firstItem);

      expect(onWordChange).toHaveBeenCalledWith('p1_l1', 'p1_l1_w1', 'Amoxicillin');
    });
  });

  describe('Speed Review Queue & Quick-Pick Shortcuts', () => {
    it('populates speed queue with low-confidence tokens (<70%) and renders quick-picks', () => {
      render(<InlineEditor page={sampleMedicalPage} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      fireEvent.click(speedTab);

      expect(screen.getByTestId('speed-review-panel')).toBeInTheDocument();
      expect(screen.getByTestId('speed-review-input')).toHaveValue('Amox');

      // Check quick-pick numbered suggestions
      expect(screen.getByTestId('speed-suggestion-1')).toBeInTheDocument();
      expect(screen.getByTestId('speed-suggestion-1')).toHaveTextContent('Amoxicillin');
    });

    it('quick-selects suggestion using number key 1 and advances to next item', () => {
      const onWordChange = vi.fn();
      render(<InlineEditor page={sampleMedicalPage} onWordChange={onWordChange} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      fireEvent.click(speedTab);

      const input = screen.getByTestId('speed-review-input');
      // Press '1'
      fireEvent.keyDown(input, { key: '1' });

      expect(onWordChange).toHaveBeenCalledWith('p1_l1', 'p1_l1_w1', 'Amoxicillin');
      // Should advance to second item ("500")
      expect(screen.getByTestId('speed-review-input')).toHaveValue('500');
    });

    it('navigates with Prev and Skip buttons', () => {
      render(<InlineEditor page={sampleMedicalPage} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      fireEvent.click(speedTab);

      expect(screen.getByTestId('speed-review-input')).toHaveValue('Amox');

      // Click Skip
      fireEvent.click(screen.getByTestId('btn-speed-skip'));
      expect(screen.getByTestId('speed-review-input')).toHaveValue('500');

      // Click Prev
      fireEvent.click(screen.getByTestId('btn-speed-prev'));
      expect(screen.getByTestId('speed-review-input')).toHaveValue('Amox');
    });

    it('displays completion banner when no low confidence tokens exist', () => {
      const allHighConfPage: PageResult = {
        page_number: 1,
        width: 800,
        height: 1000,
        full_text: 'Lisinopril 10mg PO QD',
        mean_confidence: 0.98,
        lines: [
          {
            line_id: 'l1',
            text: 'Lisinopril 10mg PO QD',
            confidence: 0.98,
            bbox: [0.1, 0.1, 0.2, 0.9],
            words: [
              { word_id: 'w1', text: 'Lisinopril', confidence: 0.99, bbox: [0.1, 0.1, 0.2, 0.4] },
              { word_id: 'w2', text: '10mg', confidence: 0.98, bbox: [0.1, 0.45, 0.2, 0.6] },
            ],
          },
        ],
      };

      render(<InlineEditor page={allHighConfPage} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      fireEvent.click(speedTab);

      expect(screen.getByText('All Words Verified!')).toBeInTheDocument();
      expect(screen.getByTestId('btn-return-lines')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('btn-return-lines'));
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });
  });
});
