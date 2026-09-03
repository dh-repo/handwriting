/**
 * frontend/src/__tests__/challenger_m4_rem1_stress.test.tsx
 * Empirical Challenger Stress & Adversarial Test Suite for M4 Remediation.
 *
 * Objectives:
 * 1. Empirically verify typing numbers like "500mg", "10mg", "1 tab" in `speed-review-input`
 *    in InlineEditor.tsx:
 *    - Verify behavior while operator is actively typing (speedInputTouched = true).
 *    - Verify behavior on initial untouched focus (quick-pick triage).
 *    - Verify behavior with Alt+1..5 modifier.
 *    - Verify behavior on empty input / clearing.
 * 2. Empirically verify rapid debouncing (500ms trailing edge) and immediate dispatch on Enter and onBlur.
 * 3. Empirically verify Revert All button timer cancellation and state reset.
 */

import React from 'react';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult, FeedbackSubmissionRequest, FeedbackSubmissionResponse, LineItem } from '@/types/ocr';
import { ApiClient } from '@/lib/apiClient';

describe('Challenger M4 Remediation Stress Harness', () => {
  const createMockPage = (): PageResult => ({
    page_number: 1,
    width: 800,
    height: 1000,
    image_url: 'blob:test-m4-rem',
    mean_confidence: 0.50,
    full_text: 'Rx: Amox 500mg daily\nTake 1 tab q12h',
    lines: [
      {
        line_id: 'line_1',
        text: 'Rx: Amox 500mg daily',
        original_text: 'Rx: Amox 500mg daily',
        confidence: 0.52,
        bbox: [0.1, 0.1, 0.18, 0.9],
        words: [
          {
            word_id: 'w_rx',
            text: 'Rx:',
            original_text: 'Rx:',
            confidence: 0.95,
            bbox: [0.1, 0.1, 0.18, 0.2],
          },
          {
            word_id: 'w_amox',
            text: 'Amox',
            original_text: 'Amox',
            confidence: 0.45,
            bbox: [0.1, 0.22, 0.18, 0.45],
          },
          {
            word_id: 'w_dose',
            text: '500mg',
            original_text: '500mg',
            confidence: 0.48,
            bbox: [0.1, 0.48, 0.18, 0.7],
          },
          {
            word_id: 'w_daily',
            text: 'daily',
            original_text: 'daily',
            confidence: 0.92,
            bbox: [0.1, 0.72, 0.18, 0.9],
          },
        ],
      },
      {
        line_id: 'line_2',
        text: 'Take 1 tab q12h',
        original_text: 'Take 1 tab q12h',
        confidence: 0.50,
        bbox: [0.2, 0.1, 0.28, 0.9],
        words: [
          {
            word_id: 'w_take',
            text: 'Take',
            original_text: 'Take',
            confidence: 0.90,
            bbox: [0.2, 0.1, 0.28, 0.3],
          },
          {
            word_id: 'w_1',
            text: '1',
            original_text: '1',
            confidence: 0.42,
            bbox: [0.2, 0.32, 0.28, 0.4],
          },
          {
            word_id: 'w_tab',
            text: 'tab',
            original_text: 'tab',
            confidence: 0.44,
            bbox: [0.2, 0.42, 0.28, 0.6],
          },
          {
            word_id: 'w_q12h',
            text: 'q12h',
            original_text: 'q12h',
            confidence: 0.88,
            bbox: [0.2, 0.62, 0.28, 0.9],
          },
        ],
      },
    ],
  });

  let mockSubmitFeedback: ReturnType<typeof vi.fn>;
  let mockClient: ApiClient;

  beforeEach(() => {
    mockSubmitFeedback = vi.fn().mockImplementation(async (req: FeedbackSubmissionRequest) => {
      return {
        status: 'persisted',
        feedback_id: `fb_rem1_${Date.now()}`,
        document_id: req.document_id,
        line_id: req.line_id,
        manifest_path: 'data/feedback/manifest.jsonl',
        confusion_pairs_count: 1,
        timestamp: req.timestamp || new Date().toISOString(),
      } as FeedbackSubmissionResponse;
    });

    mockClient = {
      submitFeedback: mockSubmitFeedback,
    } as unknown as ApiClient;
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  // ==========================================================================
  // SECTION 1: SPEED REVIEW INPUT NUMBER TYPING STRESS ("500mg", "10mg", "1 tab")
  // ==========================================================================
  describe('Speed Review Input: Numerical Typing & Shortcut Isolation', () => {
    it('allows operator to actively type "500mg" without key 5 hijacking input', async () => {
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_500mg"
          apiClientInstance={mockClient}
        />
      );

      // Navigate to Speed Review tab
      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

      // Operator begins active typing: changes input value
      fireEvent.change(speedInput, { target: { value: '50' } });
      // Operator presses key '0'
      fireEvent.keyDown(speedInput, { key: '0' });
      // Operator presses key '0'
      fireEvent.keyDown(speedInput, { key: '0' });
      // Operator presses key '5' while actively typing
      fireEvent.keyDown(speedInput, { key: '5' });

      // Ensure keydown '5' was NOT hijacked (no submission yet)
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Complete full dosage string "500mg"
      fireEvent.change(speedInput, { target: { value: '500mg' } });
      // Press Enter to submit operator's custom correction
      fireEvent.keyDown(speedInput, { key: 'Enter', code: 'Enter' });

      await act(async () => {
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const callPayload = mockSubmitFeedback.mock.calls[0][0];
      expect(callPayload.operator_correction).toBe('500mg');
      expect(callPayload.original_prediction).toBe('Amox');
    });

    it('allows operator to actively type "10mg" without key 1 hijacking input', async () => {
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_10mg"
          apiClientInstance={mockClient}
        />
      );

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

      // Operator types "10mg"
      fireEvent.change(speedInput, { target: { value: '1' } });
      // Operator presses key 1 while actively typing
      fireEvent.keyDown(speedInput, { key: '1' });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      fireEvent.change(speedInput, { target: { value: '10mg' } });
      fireEvent.keyDown(speedInput, { key: 'Enter', code: 'Enter' });

      await act(async () => {
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const callPayload = mockSubmitFeedback.mock.calls[0][0];
      expect(callPayload.operator_correction).toBe('10mg');
      expect(callPayload.original_prediction).toBe('Amox');
    });

    it('allows operator to actively type "1 tab" without key 1 hijacking input', async () => {
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_1tab"
          apiClientInstance={mockClient}
        />
      );

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

      // Operator types "1 tab"
      fireEvent.change(speedInput, { target: { value: '1 ' } });
      // Operator presses key 1, 2, 3, 4, 5 while typing
      fireEvent.keyDown(speedInput, { key: '1' });
      fireEvent.keyDown(speedInput, { key: '2' });
      fireEvent.keyDown(speedInput, { key: '3' });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      fireEvent.change(speedInput, { target: { value: '1 tab' } });
      fireEvent.keyDown(speedInput, { key: 'Enter', code: 'Enter' });

      await act(async () => {
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const callPayload = mockSubmitFeedback.mock.calls[0][0];
      expect(callPayload.operator_correction).toBe('1 tab');
      expect(callPayload.original_prediction).toBe('Amox');
    });

    it('allows Alt+1..5 to explicitly select suggestions even while typing', async () => {
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_alt_select"
          apiClientInstance={mockClient}
        />
      );

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

      // Operator edits input
      fireEvent.change(speedInput, { target: { value: 'partial edit' } });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Operator presses Alt+1 to force selection of top suggestion
      fireEvent.keyDown(speedInput, { key: '1', altKey: true });

      await act(async () => {
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const callPayload = mockSubmitFeedback.mock.calls[0][0];
      expect(callPayload.operator_correction).toMatch(/Amoxicillin/i);
    });

    it('investigates behavior when operator clears input to empty string and types number 1-5', async () => {
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_empty_then_num"
          apiClientInstance={mockClient}
        />
      );

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

      // Operator clears the input completely
      fireEvent.change(speedInput, { target: { value: '' } });

      // If user types '1' while input is empty, suggestion 1 is selected (untouched or empty triage)
      // Whereas when user is actively typing a value like "10mg" or "1 tab", digits do NOT hijack
      fireEvent.keyDown(speedInput, { key: '1' });

      await act(async () => {
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      expect(mockSubmitFeedback.mock.calls[0][0].operator_correction).toMatch(/Amoxicillin/i);
    });

    it('triggers quick-pick when operator presses 1-5 on untouched initial triage', async () => {
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_untouched_triage"
          apiClientInstance={mockClient}
        />
      );

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

      // Untouched input: operator presses '1' immediately to triage
      fireEvent.keyDown(speedInput, { key: '1' });

      await act(async () => {
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const callPayload = mockSubmitFeedback.mock.calls[0][0];
      expect(callPayload.operator_correction).toMatch(/Amoxicillin/i);
    });
  });

  // ==========================================================================
  // SECTION 2: 500MS TRAILING-EDGE DEBOUNCING AND IMMEDIATE ENTER / ONBLUR
  // ==========================================================================
  describe('Debounced Line Feedback Dispatch (500ms Trailing Edge, Enter, Blur)', () => {
    it('debounces rapid typing with exactly 500ms trailing-edge delay', async () => {
      vi.useFakeTimers();
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_debounce"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_1');

      // 10 rapid keystrokes arriving every 50ms (total 500ms elapsed, but timer resets each time)
      for (let i = 1; i <= 10; i++) {
        act(() => {
          fireEvent.change(input, { target: { value: `Debounce Test Edit ${i}` } });
          vi.advanceTimersByTime(50);
        });
        expect(mockSubmitFeedback).not.toHaveBeenCalled();
        expect(screen.getByTestId('feedback-status-line_1')).toHaveTextContent(/Staged/i);
      }

      // 490ms after last keystroke: still not dispatched
      act(() => {
        vi.advanceTimersByTime(490);
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Pass 500ms boundary
      await act(async () => {
        vi.advanceTimersByTime(20);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      expect(mockSubmitFeedback.mock.calls[0][0].operator_correction).toBe('Debounce Test Edit 10');
      expect(screen.getByTestId('feedback-status-line_1')).toHaveTextContent(/Flywheel Saved/i);
    });

    it('immediately dispatches on Enter and cancels pending 500ms debounce timer with zero duplicates', async () => {
      vi.useFakeTimers();
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_enter"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_1');

      act(() => {
        fireEvent.change(input, { target: { value: 'Immediate Enter Correction' } });
        vi.advanceTimersByTime(120); // 120ms into 500ms timer
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Press Enter
      await act(async () => {
        fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      expect(mockSubmitFeedback.mock.calls[0][0].operator_correction).toBe('Immediate Enter Correction');

      // Fast forward past the original 500ms debounce timer
      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      // Must remain exactly 1 call (no double-flush)
      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    });

    it('immediately flushes on onBlur if debounce timer is active and cancels pending timer', async () => {
      vi.useFakeTimers();
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_blur"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_1');

      act(() => {
        fireEvent.change(input, { target: { value: 'Immediate Blur Correction' } });
        vi.advanceTimersByTime(200); // 200ms into 500ms timer
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Blur input
      await act(async () => {
        fireEvent.blur(input);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      expect(mockSubmitFeedback.mock.calls[0][0].operator_correction).toBe('Immediate Blur Correction');

      // Fast forward past 500ms timer
      await act(async () => {
        vi.advanceTimersByTime(800);
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    });

    it('does not re-dispatch on onBlur if already synced via Enter or prior debounce timer', async () => {
      vi.useFakeTimers();
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_no_double_blur"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_1');

      // Edit and press Enter
      await act(async () => {
        fireEvent.change(input, { target: { value: 'Enter Synced Value' } });
        fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);

      // Now blur the input
      await act(async () => {
        fireEvent.blur(input);
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      // Blur should NOT trigger a second feedback dispatch
      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    });

    it('validates canonical M1 ↔ M4 schema contract fidelity on dispatched payloads', async () => {
      vi.useFakeTimers();
      render(
        <InlineEditor
          page={createMockPage()}
          documentId="doc_stress_contract_payload"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_1');

      await act(async () => {
        fireEvent.change(input, { target: { value: 'Rx: Amoxicillin 500mg daily' } });
        fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const payload: FeedbackSubmissionRequest = mockSubmitFeedback.mock.calls[0][0];

      // M1 Schema Mandatory Contract Fields
      expect(payload.document_id).toBe('doc_stress_contract_payload');
      expect(payload.page_number).toBe(1);
      expect(payload.line_id).toBe('line_1');
      expect(payload.original_prediction).toBe('Rx: Amox 500mg daily');
      expect(payload.operator_correction).toBe('Rx: Amoxicillin 500mg daily');

      // Legacy Backward-Compatible Aliases
      expect(payload.original_text).toBe('Rx: Amox 500mg daily');
      expect(payload.corrected_text).toBe('Rx: Amoxicillin 500mg daily');

      // Metadata fidelity
      expect(payload.confidence).toBe(0.52);
      expect(payload.bbox).toEqual([0.1, 0.1, 0.18, 0.9]);
      expect(typeof payload.timestamp).toBe('string');
      expect(new Date(payload.timestamp!).getTime()).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // SECTION 3: REVERT ALL BUTTON TIMER CANCELLATION
  // ==========================================================================
  describe('Revert All: Debounce Timer Cancellation & State Reset', () => {
    it('cancels all active line debounce timers when Revert All is clicked', async () => {
      vi.useFakeTimers();
      const onRevertAll = vi.fn();
      const page = createMockPage();
      page.lines[0].is_edited = true;

      render(
        <InlineEditor
          page={page}
          documentId="doc_stress_revert_all"
          apiClientInstance={mockClient}
          onRevertAll={onRevertAll}
        />
      );

      const input1 = screen.getByTestId('line-input-line_1');
      const input2 = screen.getByTestId('line-input-line_2');

      // Start pending debounce timer on line 1
      act(() => {
        fireEvent.change(input1, { target: { value: 'Unsaved Edit Line 1' } });
        vi.advanceTimersByTime(100);
      });
      expect(screen.getByTestId('feedback-status-line_1')).toHaveTextContent(/Staged/i);

      // Start pending debounce timer on line 2
      act(() => {
        fireEvent.change(input2, { target: { value: 'Unsaved Edit Line 2' } });
        vi.advanceTimersByTime(100);
      });
      expect(screen.getByTestId('feedback-status-line_2')).toHaveTextContent(/Staged/i);

      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Find and click Revert All button
      const revertBtn =
        screen.queryByTestId('btn-revert-all') ||
        screen.queryByTestId('revert-all-btn');
      expect(revertBtn).not.toBeNull();

      act(() => {
        fireEvent.click(revertBtn!);
      });

      expect(onRevertAll).toHaveBeenCalledTimes(1);

      // Fast-forward past all debounce timers (e.g. 5000ms)
      await act(async () => {
        vi.advanceTimersByTime(5000);
        await Promise.resolve();
        await Promise.resolve();
      });

      // ZERO feedback dispatches should occur because timers were aborted!
      expect(mockSubmitFeedback).toHaveBeenCalledTimes(0);

      // Status badges should be cleared
      expect(screen.queryByTestId('feedback-status-line_1')).toBeNull();
      expect(screen.queryByTestId('feedback-status-line_2')).toBeNull();
    });
  });
});
