import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ZeroFrictionSampleCards } from '@/components/ZeroFrictionSampleCards';

describe('ZeroFrictionSampleCards Component', () => {
  it('renders 3 sample cards with titles, accuracy metrics, and descriptions', () => {
    render(<ZeroFrictionSampleCards onSelectSample={() => {}} />);

    expect(screen.getByTestId('zero-friction-samples-container')).toBeInTheDocument();

    // 1. 18th-century cursive
    expect(screen.getByText(/18th-Century Cursive/i)).toBeInTheDocument();
    expect(screen.getByText(/96% accuracy/i)).toBeInTheDocument();

    // 2. Annotated meeting notes
    expect(screen.getByText(/Annotated Meeting Notes/i)).toBeInTheDocument();
    expect(screen.getByText(/95% accuracy/i)).toBeInTheDocument();

    // 3. Messy receipt & clinical rx
    expect(screen.getByText(/Messy Receipt & Clinical Rx/i)).toBeInTheDocument();
    expect(screen.getByText(/94% accuracy/i)).toBeInTheDocument();
  });

  it('triggers onSelectSample callback with preset data when card is clicked', () => {
    const onSelectSample = vi.fn();
    render(<ZeroFrictionSampleCards onSelectSample={onSelectSample} />);

    const card = screen.getByTestId('sample-card-sample_18th_century');
    fireEvent.click(card);

    expect(onSelectSample).toHaveBeenCalled();
    const calledDoc = onSelectSample.mock.calls[0][0];
    expect(calledDoc.filename).toBe('sample_clean_cursive.png');
    expect(calledDoc.pages.length).toBeGreaterThanOrEqual(1);
  });
});
