import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Dropzone } from '@/components/Dropzone';

describe('Dropzone Component', () => {
  it('renders upload instructions and format badges', () => {
    render(<Dropzone onFileAccepted={() => {}} />);
    expect(
      screen.getByText(/Drop your handwriting image or PDF here/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/PNG, JPEG, TIFF/i)).toBeInTheDocument();
    expect(screen.getByText(/Multi-page PDF/i)).toBeInTheDocument();
  });

  it('accepts valid PNG image file drop', () => {
    const onFileAccepted = vi.fn();
    render(<Dropzone onFileAccepted={onFileAccepted} />);

    const dropzone = screen.getByTestId('dropzone-container');
    const file = new File(['dummy-content'], 'test_cursive.png', { type: 'image/png' });

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
      },
    });

    expect(onFileAccepted).toHaveBeenCalledWith(file);
  });

  it('displays error message for unsupported file formats', () => {
    const onFileAccepted = vi.fn();
    render(<Dropzone onFileAccepted={onFileAccepted} />);

    const dropzone = screen.getByTestId('dropzone-container');
    const file = new File(['dummy-content'], 'archive.zip', { type: 'application/zip' });

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
      },
    });

    expect(onFileAccepted).not.toHaveBeenCalled();
    expect(screen.getByTestId('dropzone-error')).toHaveTextContent(/Unsupported file format/i);
  });

  it('renders loading spinner and progress bar during processing', () => {
    render(
      <Dropzone
        onFileAccepted={() => {}}
        isLoading={true}
        uploadProgress={65}
        processingStage="Segmenting handwriting lines..."
      />
    );

    expect(screen.getByText(/Transcribing Handwriting.../i)).toBeInTheDocument();
    expect(screen.getByText(/Segmenting handwriting lines.../i)).toBeInTheDocument();
  });
});
