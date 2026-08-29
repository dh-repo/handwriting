import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { MetricsSummary } from '@/components/MetricsSummary';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MESSY_CURSIVE } from '@/lib/sampleDocuments';

describe('MetricsSummary Component', () => {
  it('displays mean confidence, total words, and line count', () => {
    render(<MetricsSummary document={SAMPLE_CLEAN_CURSIVE} />);

    expect(screen.getByTestId('metrics-summary-container')).toBeInTheDocument();
    expect(screen.getByTestId('metric-mean-confidence')).toHaveTextContent('96.2%');
    expect(screen.getByTestId('metric-total-words')).toHaveTextContent('28');
    expect(screen.getByTestId('metric-total-lines')).toHaveTextContent('4');
    expect(screen.getByTestId('metric-latency')).toHaveTextContent('184 ms');
  });

  it('displays low-confidence warning count and triggers speed review callback', () => {
    const onStartSpeedReview = vi.fn();
    render(
      <MetricsSummary
        document={SAMPLE_MESSY_CURSIVE}
        onStartSpeedReview={onStartSpeedReview}
      />
    );

    expect(screen.getByTestId('metric-low-conf-count')).not.toHaveTextContent('0');

    const reviewBtn = screen.getByTestId('btn-metric-speed-review');
    fireEvent.click(reviewBtn);
    expect(onStartSpeedReview).toHaveBeenCalled();
  });
});
