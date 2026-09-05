import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ZeroFrictionSampleCards } from '@/components/ZeroFrictionSampleCards';

describe('ZeroFrictionSampleCards Component', () => {
  it('labels prepared general examples without accuracy claims', () => {
    render(<ZeroFrictionSampleCards onSelectSample={() => {}} />);
    expect(screen.getByText('Cursive Note')).toBeInTheDocument();
    expect(screen.getByText('Annotated Meeting Notes')).toBeInTheDocument();
    expect(screen.getAllByText('Prepared demo')).toHaveLength(2);
    expect(screen.queryByText(/% accuracy/)).toBeNull();
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
