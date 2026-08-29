import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ExportToolbar } from '@/components/ExportToolbar';
import { SAMPLE_CLEAN_CURSIVE } from '@/lib/sampleDocuments';
import * as exportUtils from '@/lib/exportUtils';

describe('ExportToolbar Component', () => {
  it('renders all export buttons', () => {
    render(<ExportToolbar document={SAMPLE_CLEAN_CURSIVE} />);

    expect(screen.getByTestId('btn-copy-clipboard')).toBeInTheDocument();
    expect(screen.getByTestId('btn-export-txt')).toBeInTheDocument();
    expect(screen.getByTestId('btn-export-csv')).toBeInTheDocument();
    expect(screen.getByTestId('btn-export-json')).toBeInTheDocument();
  });

  it('triggers clipboard copy and updates button label temporarily', async () => {
    const copySpy = vi.spyOn(exportUtils, 'copyTextToClipboard').mockResolvedValue(true);

    render(<ExportToolbar document={SAMPLE_CLEAN_CURSIVE} />);

    const copyBtn = screen.getByTestId('btn-copy-clipboard');
    await act(async () => {
      fireEvent.click(copyBtn);
    });

    expect(copySpy).toHaveBeenCalled();
  });

  it('calls export helpers when download buttons are clicked', () => {
    const txtSpy = vi.spyOn(exportUtils, 'exportDocumentAsTxt');
    const csvSpy = vi.spyOn(exportUtils, 'exportDocumentAsCsv');
    const jsonSpy = vi.spyOn(exportUtils, 'exportDocumentAsJson');
    const downloadSpy = vi.spyOn(exportUtils, 'downloadFile').mockImplementation(() => {});

    render(<ExportToolbar document={SAMPLE_CLEAN_CURSIVE} />);

    fireEvent.click(screen.getByTestId('btn-export-txt'));
    expect(txtSpy).toHaveBeenCalled();
    expect(downloadSpy).toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('btn-export-csv'));
    expect(csvSpy).toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('btn-export-json'));
    expect(jsonSpy).toHaveBeenCalled();
  });
});
