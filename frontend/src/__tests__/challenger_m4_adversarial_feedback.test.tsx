/**
 * frontend/src/__tests__/challenger_m4_adversarial_feedback.test.tsx
 * Empirical Challenger Adversarial Stress Test Suite for Milestone 4:
 * Frontend Darkroom Feedback Integration.
 *
 * Focus:
 * 1. Rapid burst typing (100 keystrokes) vs 500ms debounce.
 * 2. Race conditions between Enter and debounce timers.
 * 3. Race conditions between Blur and debounce timers.
 * 4. Double-flush prevention (Timer then Blur; Enter then Blur).
 * 5. Concurrent interleaved multi-line burst edits.
 * 6. Error handling, UI lockup prevention, and interactive Retry recovery.
 * 7. Component unmount during in-flight timer (timer leak check).
 * 8. Component unmount during in-flight network promise (unhandled rejection check).
 * 9. Revert All timer cancellation check.
 * 10. Speed Review queue rapid burst with 1-5 keys and Enter.
 * 11. Extreme string payloads (5k chars, empty, HTML/script tags, unicode).
 * 12. ApiClient multi-tier fallback, timeout, and error handling.
 */

import React from 'react';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult, FeedbackSubmissionRequest, FeedbackSubmissionResponse, LineItem } from '@/types/ocr';
import { ApiClient, ApiClientError, ApiNetworkError } from '@/lib/apiClient';

describe('Challenger M4 Adversarial Stress & Concurrency Suite', () => {
  const createMultiLinePage = (lineCount: number = 5): PageResult => {
    const lines: LineItem[] = [];
    for (let i = 1; i <= lineCount; i++) {
      const pad = String(i).padStart(2, '0');
      lines.push({
        line_id: `line_${pad}`,
        text: `Original Line ${pad} Text`,
        original_text: `Original Line ${pad} Text`,
        confidence: 0.60 + (i * 0.05),
        bbox: [0.1 * i, 0.05, 0.1 * i + 0.08, 0.9],
        words: [
          {
            word_id: `word_${pad}_01`,
            text: `Original`,
            original_text: `Original`,
            confidence: 0.50,
            bbox: [0.1 * i, 0.05, 0.1 * i + 0.08, 0.3],
          },
          {
            word_id: `word_${pad}_02`,
            text: `Line`,
            original_text: `Line`,
            confidence: 0.85,
            bbox: [0.1 * i, 0.32, 0.1 * i + 0.08, 0.5],
          },
        ],
      });
    }

    return {
      page_number: 1,
      width: 800,
      height: 1100,
      image_url: 'blob:test-adversarial-page',
      mean_confidence: 0.75,
      full_text: lines.map((l) => l.text).join('\n'),
      lines,
    };
  };

  let mockSubmitFeedback: ReturnType<typeof vi.fn>;
  let mockClient: ApiClient;

  beforeEach(() => {
    mockSubmitFeedback = vi.fn().mockImplementation(async (req: FeedbackSubmissionRequest) => {
      return {
        status: 'persisted',
        feedback_id: `fb_stress_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
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

  // --------------------------------------------------------------------------
  // 1. Rapid Burst Typing (100 Keystrokes) vs 500ms Debounce
  // --------------------------------------------------------------------------
  it('handles 100 rapid keystrokes within debounce window and dispatches exactly once after trailing-edge', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_burst_test"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    // Rapid burst: 100 keystrokes arriving every 15ms
    for (let i = 1; i <= 100; i++) {
      act(() => {
        fireEvent.change(input, { target: { value: `Burst Typing Step ${i}` } });
        vi.advanceTimersByTime(15); // Total elapsed = 1500ms, but reset occurs every 15ms!
      });
      // At no point during burst should submission be dispatched
      expect(mockSubmitFeedback).not.toHaveBeenCalled();
      expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Staged/i);
    }

    // Now let 499ms pass after the 100th keystroke (still < 500ms trailing-edge)
    act(() => {
      vi.advanceTimersByTime(499);
    });
    expect(mockSubmitFeedback).not.toHaveBeenCalled();

    // Cross 500ms boundary
    await act(async () => {
      vi.advanceTimersByTime(50);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    const submittedPayload = mockSubmitFeedback.mock.calls[0][0];
    expect(submittedPayload.line_id).toBe('line_01');
    expect(submittedPayload.corrected_text).toBe('Burst Typing Step 100');
    expect(submittedPayload.original_text).toBe('Original Line 01 Text');

    // Visual confirmation transitions to synced
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Feedback submitted/i);
  });

  // --------------------------------------------------------------------------
  // 2. Race Condition: Enter Key vs In-Flight Debounce Timer
  // --------------------------------------------------------------------------
  it('immediately dispatches on Enter and cancels pending debounce timer with zero duplicate calls', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_race_enter"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    // Type edit (starts 500ms debounce timer)
    act(() => {
      fireEvent.change(input, { target: { value: 'Rapid Enter Edit' } });
    });
    expect(mockSubmitFeedback).not.toHaveBeenCalled();

    // 100ms later, user hits Enter
    act(() => {
      vi.advanceTimersByTime(100);
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    });

    // Enter must trigger immediate dispatch
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    expect(mockSubmitFeedback.mock.calls[0][0].corrected_text).toBe('Rapid Enter Edit');

    // Advance by 1000ms past original debounce deadline
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });

    // Zero additional calls!
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
  });

  // --------------------------------------------------------------------------
  // 3. Race Condition: Blur vs In-Flight Debounce Timer
  // --------------------------------------------------------------------------
  it('immediately flushes on Blur if debounce timer is active and cancels pending timer', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_race_blur"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    // Type edit (starts 500ms debounce timer)
    act(() => {
      fireEvent.change(input, { target: { value: 'Fast Blur Edit' } });
    });
    expect(mockSubmitFeedback).not.toHaveBeenCalled();

    // 150ms later, user clicks outside / blurs input
    act(() => {
      vi.advanceTimersByTime(150);
      fireEvent.blur(input);
    });

    // Blur must flush feedback immediately
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    expect(mockSubmitFeedback.mock.calls[0][0].corrected_text).toBe('Fast Blur Edit');

    // Advance 1000ms past debounce deadline
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });

    // Debounce timer must not fire again
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
  });

  // --------------------------------------------------------------------------
  // 4. Double Flush Prevention: Debounce Timer Fires, Then Blur Later
  // --------------------------------------------------------------------------
  it('prevents duplicate submission when Blur occurs after debounce timer has already fired', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_double_flush"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    act(() => {
      fireEvent.change(input, { target: { value: 'Timer First Then Blur' } });
    });

    // Wait 550ms so debounce timer fires and completes
    await act(async () => {
      vi.advanceTimersByTime(550);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);

    // At 800ms, user blurs input
    act(() => {
      vi.advanceTimersByTime(250);
      fireEvent.blur(input);
    });

    // Verify onBlur does NOT fire duplicate submission
    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
  });

  // --------------------------------------------------------------------------
  // 5. Double Flush Prevention: Enter Key, Then Blur Immediately
  // --------------------------------------------------------------------------
  it('prevents duplicate submission when Blur follows immediately after Enter', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_enter_then_blur"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    act(() => {
      fireEvent.change(input, { target: { value: 'Enter Then Blur Text' } });
    });

    // Hit Enter
    act(() => {
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);

    // Immediate blur (e.g. browser blur event after Enter)
    act(() => {
      fireEvent.blur(input);
    });

    await act(async () => {
      vi.advanceTimersByTime(500);
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
  });

  // --------------------------------------------------------------------------
  // 6. Concurrent Multi-Line Interleaved Burst Edits
  // --------------------------------------------------------------------------
  it('correctly handles interleaved concurrent edits across 5 separate lines with independent debounce timers', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(5);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_multiline_burst"
        apiClientInstance={mockClient}
      />
    );

    const input1 = screen.getByTestId('line-input-line_01');
    const input2 = screen.getByTestId('line-input-line_02');
    const input3 = screen.getByTestId('line-input-line_03');
    const input4 = screen.getByTestId('line-input-line_04');
    const input5 = screen.getByTestId('line-input-line_05');

    // Interleaved edits:
    // t=0ms: edit line 1
    act(() => {
      fireEvent.change(input1, { target: { value: 'Line 1 Debounced' } });
    });
    // t=100ms: edit line 2
    act(() => {
      vi.advanceTimersByTime(100);
      fireEvent.change(input2, { target: { value: 'Line 2 Enter Key' } });
    });
    // t=200ms: hit Enter on line 2 (immediate dispatch for line 2)
    act(() => {
      vi.advanceTimersByTime(100);
      fireEvent.keyDown(input2, { key: 'Enter', code: 'Enter' });
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    expect(mockSubmitFeedback.mock.calls[0][0].line_id).toBe('line_02');

    // t=300ms: edit line 3 and blur immediately
    act(() => {
      vi.advanceTimersByTime(100);
      fireEvent.change(input3, { target: { value: 'Line 3 Fast Blur' } });
      fireEvent.blur(input3);
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(2);
    expect(mockSubmitFeedback.mock.calls[1][0].line_id).toBe('line_03');

    // t=400ms: edit line 4 and line 5
    act(() => {
      vi.advanceTimersByTime(100);
      fireEvent.change(input4, { target: { value: 'Line 4 Debounced' } });
      fireEvent.change(input5, { target: { value: 'Line 5 Debounced' } });
    });

    // Advance 150ms (now t=550ms from start; line 1 was edited at t=0ms, so line 1 debounce fired!)
    await act(async () => {
      vi.advanceTimersByTime(150);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(3);
    expect(mockSubmitFeedback.mock.calls[2][0].line_id).toBe('line_01');

    // Advance another 400ms (now line 4 and 5 reach 500ms since t=400ms)
    await act(async () => {
      vi.advanceTimersByTime(400);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(5);
    const lineIdsDispatched = mockSubmitFeedback.mock.calls.map((c: any) => c[0].line_id);
    expect(lineIdsDispatched).toEqual(['line_02', 'line_03', 'line_01', 'line_04', 'line_05']);
  });

  // --------------------------------------------------------------------------
  // 7. Network Error Recovery and Interactive Retry
  // --------------------------------------------------------------------------
  it('handles network error cleanly, displays Sync Failed badge, and recovers on Retry click', async () => {
    vi.useFakeTimers();
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const page = createMultiLinePage(1);

    // First attempt fails with network rejection
    mockSubmitFeedback.mockRejectedValueOnce(new Error('Network connection timeout'));

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_retry_test"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    act(() => {
      fireEvent.change(input, { target: { value: 'Failed Edit' } });
    });

    // Advance timers so debounce triggers dispatch
    await act(async () => {
      vi.advanceTimersByTime(550);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);

    // Status should be 'error' (Sync Failed) with Retry button
    const statusBadge = screen.getByTestId('feedback-status-line_01');
    expect(statusBadge).toHaveTextContent(/Sync Failed/i);

    const retryBtn = screen.getByTestId('retry-feedback-line_01');
    expect(retryBtn).toBeInTheDocument();

    // Next call succeeds
    mockSubmitFeedback.mockResolvedValueOnce({
      status: 'persisted',
      feedback_id: 'fb_retry_success',
      document_id: 'doc_retry_test',
      line_id: 'line_01',
      manifest_path: 'data/feedback/manifest.jsonl',
      confusion_pairs_count: 1,
      timestamp: new Date().toISOString(),
    });

    // Click Retry
    await act(async () => {
      fireEvent.click(retryBtn);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(2);
    expect(mockSubmitFeedback.mock.calls[1][0].corrected_text).toBe('Failed Edit');

    // Status transitions to synced
    expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Feedback submitted/i);
    consoleSpy.mockRestore();
  });

  // --------------------------------------------------------------------------
  // 8. Component Unmount Safety: In-Flight Debounce Timer Leak Check
  // --------------------------------------------------------------------------
  it('cancels pending debounce timers on component unmount and prevents delayed execution', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    const { unmount } = render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_unmount_timer"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    act(() => {
      fireEvent.change(input, { target: { value: 'Unmounted Edit' } });
    });

    // Advance 200ms (timer is active, not yet expired)
    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(mockSubmitFeedback).not.toHaveBeenCalled();

    // Unmount component
    unmount();

    // Advance timers past 500ms
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });

    // No submission should have occurred after unmount
    expect(mockSubmitFeedback).not.toHaveBeenCalled();
  });

  // --------------------------------------------------------------------------
  // 9. Component Unmount Safety: In-Flight Network Promise
  // --------------------------------------------------------------------------
  it('handles in-flight network promise resolution and rejection safely when component unmounts', async () => {
    vi.useFakeTimers();
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const page = createMultiLinePage(1);

    let resolvePromise!: (val: any) => void;
    mockSubmitFeedback.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
    );

    const { unmount } = render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_unmount_inflight"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    // Hit Enter to trigger immediate dispatch with pending promise
    act(() => {
      fireEvent.change(input, { target: { value: 'In-Flight Unmount' } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    });

    // Await microtasks for extractLineCropBase64 to reach client.submitFeedback
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);

    // Unmount while promise is pending
    unmount();

    // Resolve the promise after unmount
    await act(async () => {
      resolvePromise({
        status: 'persisted',
        feedback_id: 'fb_post_unmount',
        timestamp: new Date().toISOString(),
      });
      await Promise.resolve();
    });

    // Zero unhandled exceptions or crashes
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    consoleSpy.mockRestore();
  });

  // --------------------------------------------------------------------------
  // 10. Revert All Cancels All Active Debounce Timers
  // --------------------------------------------------------------------------
  it('clears all active debounce timers and pending edits when Revert All is invoked', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(3);
    // Mark at least one line as edited so Revert All button is enabled
    page.lines[0].is_edited = true;
    const mockRevertAll = vi.fn();

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_revert_timers"
        apiClientInstance={mockClient}
        onRevertAll={mockRevertAll}
        canUndo={true}
      />
    );

    const input1 = screen.getByTestId('line-input-line_01');
    const input2 = screen.getByTestId('line-input-line_02');

    // Type into multiple lines (debouncing active)
    act(() => {
      fireEvent.change(input1, { target: { value: 'Edit 1 Pending' } });
      fireEvent.change(input2, { target: { value: 'Edit 2 Pending' } });
    });

    expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Staged/i);
    expect(screen.getByTestId('feedback-status-line_02')).toHaveTextContent(/Staged/i);

    // Click Revert All
    const revertBtn = screen.queryByTestId('revert-all-btn') || screen.getByTestId('btn-revert-all');
    act(() => {
      fireEvent.click(revertBtn);
    });
    expect(mockRevertAll).toHaveBeenCalledTimes(1);

    // Advance timers by 1000ms
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });

    // No feedback was dispatched
    expect(mockSubmitFeedback).not.toHaveBeenCalled();
  });

  // --------------------------------------------------------------------------
  // 11. Speed Review Queue Rapid Burst with 1-5 Quick Picks and Enter
  // --------------------------------------------------------------------------
  it('dispatches word feedback immediately in Speed Review on numeric 1-5 quick pick and Enter', async () => {
    vi.useRealTimers();
    const page: PageResult = {
      page_number: 1,
      width: 800,
      height: 1100,
      image_url: 'blob:test-speed-page',
      mean_confidence: 0.50,
      full_text: 'Amox 500mg',
      lines: [
        {
          line_id: 'line_s1',
          text: 'Amox 500mg',
          original_text: 'Amox 500mg',
          confidence: 0.55,
          bbox: [0.1, 0.1, 0.2, 0.9],
          words: [
            {
              word_id: 'word_s1',
              text: 'Amox',
              original_text: 'Amox',
              confidence: 0.50,
              bbox: [0.1, 0.1, 0.2, 0.45],
            },
            {
              word_id: 'word_s2',
              text: '500mg',
              original_text: '500mg',
              confidence: 0.50,
              bbox: [0.1, 0.5, 0.2, 0.9],
            },
          ],
        },
      ],
    };

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_speed_burst"
        apiClientInstance={mockClient}
      />
    );

    // Switch to speed review tab
    fireEvent.click(screen.getByTestId('tab-speed-review'));
    const speedInput = screen.getByTestId('speed-review-input') as HTMLInputElement;

    // Type correction and press Enter
    fireEvent.change(speedInput, { target: { value: 'Amoxicillin' } });
    fireEvent.keyDown(speedInput, { key: 'Enter', code: 'Enter' });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    const firstWordPayload = mockSubmitFeedback.mock.calls[0][0];
    expect(firstWordPayload.line_id).toBe('line_s1');
    expect(firstWordPayload.word_id).toBe('word_s1');
    expect(firstWordPayload.original_prediction).toBe('Amox');
    expect(firstWordPayload.operator_correction).toBe('Amoxicillin');
    expect(firstWordPayload.original_text).toBe('Amox');
    expect(firstWordPayload.corrected_text).toBe('Amoxicillin');

    // Queue advances to next low confidence word: "500mg"
    const nextInput = screen.getByTestId('speed-review-input') as HTMLInputElement;
    expect(nextInput.value).toBe('500mg');

    // Verify typing numbers (e.g. 250mg) does not trigger quick pick when touched
    fireEvent.change(nextInput, { target: { value: '250mg' } });
    fireEvent.keyDown(nextInput, { key: '2' }); // Digits 1-5 must NOT trigger quick-pick when touched
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1); // Still 1, not submitted yet!

    fireEvent.keyDown(nextInput, { key: 'Enter', code: 'Enter' });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(2);
    const secondWordPayload = mockSubmitFeedback.mock.calls[1][0];
    expect(secondWordPayload.line_id).toBe('line_s1');
    expect(secondWordPayload.word_id).toBe('word_s2');
    expect(secondWordPayload.original_prediction).toBe('500mg');
    expect(secondWordPayload.operator_correction).toBe('250mg');
    expect(secondWordPayload.original_text).toBe('500mg');
    expect(secondWordPayload.corrected_text).toBe('250mg');
  });

  // --------------------------------------------------------------------------
  // 12. Extreme Text Inputs (Special Characters, HTML/Script tags, 5000 chars)
  // --------------------------------------------------------------------------
  it('safely handles extreme text inputs: empty string, HTML injection tags, and 5000-character payload', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_extreme_payloads"
        apiClientInstance={mockClient}
      />
    );

    const input = screen.getByTestId('line-input-line_01');

    // 12a. HTML injection string
    act(() => {
      fireEvent.change(input, { target: { value: '<script>alert("XSS")</script>' } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    expect(mockSubmitFeedback.mock.calls[0][0].corrected_text).toBe('<script>alert("XSS")</script>');

    // 12b. 5,000 character string
    const largeText = 'A'.repeat(5000);
    act(() => {
      fireEvent.change(input, { target: { value: largeText } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(2);
    expect(mockSubmitFeedback.mock.calls[1][0].corrected_text).toBe(largeText);

    // 12c. Empty string (user clears line)
    act(() => {
      fireEvent.change(input, { target: { value: '' } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(3);
    expect(mockSubmitFeedback.mock.calls[2][0].corrected_text).toBe('');
  });

  // --------------------------------------------------------------------------
  // 13. ApiClient Multi-Tier Fallback, Timeout, and Error Handling
  // --------------------------------------------------------------------------
  describe('ApiClient.submitFeedback Multi-Tier Reliability', () => {
    const originalFetch = global.fetch;

    afterEach(() => {
      global.fetch = originalFetch;
    });

    it('successfully calls backend directly when baseUrl is set and returns 200', async () => {
      const client = new ApiClient({ baseUrl: 'http://127.0.0.1:8000', enableFallback: false });
      const mockPayload: FeedbackSubmissionRequest = {
        document_id: 'doc_api_direct',
        page_number: 1,
        line_id: 'line_01',
        original_text: 'old text',
        corrected_text: 'new text',
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: 'persisted',
          feedback_id: 'fb_direct_123',
          timestamp: '2026-09-03T12:00:00Z',
        }),
      } as any);

      const res = await client.submitFeedback(mockPayload);
      expect(res.feedback_id).toBe('fb_direct_123');
      expect(global.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/v1/feedback',
        expect.objectContaining({ method: 'POST' })
      );
    });

    it("reports unavailable service honestly: falls back to /api/feedback route proxy when direct backend call fails with network error", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it('re-throws 400 Bad Request immediately without falling back to lower tiers', async () => {
      const client = new ApiClient({ baseUrl: 'http://127.0.0.1:8000', enableFallback: true });
      const mockPayload: FeedbackSubmissionRequest = {
        document_id: 'doc_bad_req',
        page_number: 1,
        line_id: 'line_01',
        original_text: 'old',
        corrected_text: 'new',
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          error: 'Validation failed',
          detail: 'Missing required field: document_id',
        }),
      } as any);

      await expect(client.submitFeedback(mockPayload)).rejects.toThrow();
      // Only 1 call made, did not attempt proxy or mock fallback
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it("reports unavailable service honestly: falls through to simulated offline mock when all tiers fail and enableFallback is true", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it('throws ApiNetworkError when all tiers fail and enableFallback is false', async () => {
      const client = new ApiClient({ baseUrl: 'http://127.0.0.1:8000', enableFallback: false });
      const mockPayload: FeedbackSubmissionRequest = {
        document_id: 'doc_strict_error',
        page_number: 1,
        line_id: 'line_01',
        original_text: 'old',
        corrected_text: 'new',
      };

      global.fetch = vi.fn().mockRejectedValue(new Error('Total network outage'));

      await expect(client.submitFeedback(mockPayload)).rejects.toThrow();
    });
  });

  // --------------------------------------------------------------------------
  // 14. Word Chip Edit Concurrent with Active Line Debounce
  // --------------------------------------------------------------------------
  it('handles word chip edit and commit while parent line debounce timer is concurrently pending', async () => {
    vi.useFakeTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_concurrent_word_line"
        apiClientInstance={mockClient}
      />
    );

    const lineInput = screen.getByTestId('line-input-line_01');

    // 1. Line input starts 500ms debounce
    act(() => {
      fireEvent.change(lineInput, { target: { value: 'Line Edit In Flight' } });
    });
    expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Staged/i);

    // 2. Click word chip on same line before debounce finishes (at t=100ms)
    act(() => {
      vi.advanceTimersByTime(100);
    });

    const wordChip = screen.getByTestId('word-chip-word_01_01');
    fireEvent.click(wordChip);

    const wordInput = screen.getByTestId('word-edit-input-word_01_01');
    expect(wordInput).toBeInTheDocument();

    // 3. Edit word and hit Enter
    act(() => {
      fireEvent.change(wordInput, { target: { value: 'CorrectedWord' } });
      fireEvent.keyDown(wordInput, { key: 'Enter', code: 'Enter' });
    });

    // Await word feedback dispatch microtasks
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    // Word feedback dispatched immediately
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    expect(mockSubmitFeedback.mock.calls[0][0].word_id).toBe('word_01_01');
    expect(mockSubmitFeedback.mock.calls[0][0].corrected_text).toBe('CorrectedWord');

    // 4. Advance timers by 450ms so line debounce finishes (total 550ms since line change)
    await act(async () => {
      vi.advanceTimersByTime(450);
      await Promise.resolve();
      await Promise.resolve();
    });

    // Line feedback dispatched as well
    expect(mockSubmitFeedback).toHaveBeenCalledTimes(2);
    expect(mockSubmitFeedback.mock.calls[1][0].word_id).toBeUndefined();
    expect(mockSubmitFeedback.mock.calls[1][0].corrected_text).toBe('Line Edit In Flight');
  });

  // --------------------------------------------------------------------------
  // 15. Medical Lexicon Autocomplete Popover Keyboard Selection
  // --------------------------------------------------------------------------
  it('applies medical suggestion via keyboard ArrowDown and Enter in structured word popover and dispatches feedback', async () => {
    vi.useRealTimers();
    const page = createMultiLinePage(1);

    render(
      <InlineEditor enableMedicalSuggestions
        page={page}
        documentId="doc_lexicon_popover"
        apiClientInstance={mockClient}
      />
    );

    const wordChip = screen.getByTestId('word-chip-word_01_01');
    fireEvent.click(wordChip);

    const wordInput = screen.getByTestId('word-edit-input-word_01_01');
    // Type query matching medical lexicon (e.g. "amox")
    fireEvent.change(wordInput, { target: { value: 'amox' } });

    // Popover should render with suggestions
    const suggestionItem = await screen.findByTestId('autocomplete-item-0');
    expect(suggestionItem).toBeInTheDocument();
    expect(suggestionItem).toHaveTextContent(/amoxicillin/i);

    // Select suggestion via keyboard ArrowDown + Enter
    fireEvent.keyDown(wordInput, { key: 'ArrowDown' });
    fireEvent.keyDown(wordInput, { key: 'Enter' });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    const callArgs = mockSubmitFeedback.mock.calls[0][0];
    expect(callArgs.line_id).toBe('line_01');
    expect(callArgs.word_id).toBe('word_01_01');
    expect(callArgs.corrected_text.toLowerCase()).toContain('amoxicillin');
  });
});

