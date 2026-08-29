import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { SampleGallery, SAMPLE_CARD_CONFIGS } from '@/components/SampleGallery';

describe('SampleGallery Component', () => {
  it('renders all 4 sample preset cards', () => {
    render(<SampleGallery onSelectSample={() => {}} />);

    expect(screen.getByTestId('sample-gallery-container')).toBeInTheDocument();

    SAMPLE_CARD_CONFIGS.forEach((config) => {
      expect(screen.getByTestId(`sample-card-${config.id}`)).toBeInTheDocument();
      expect(screen.getByText(config.title)).toBeInTheDocument();
    });
  });

  it('triggers onSelectSample callback with selected preset data when clicked', () => {
    const onSelectSample = vi.fn();
    render(<SampleGallery onSelectSample={onSelectSample} />);

    const cursiveCard = screen.getByTestId('sample-card-sample_clean_cursive');
    fireEvent.click(cursiveCard);

    expect(onSelectSample).toHaveBeenCalled();
    const calledArg = onSelectSample.mock.calls[0][0];
    expect(calledArg.document_id).toBe('sample_clean_cursive');
  });
});
