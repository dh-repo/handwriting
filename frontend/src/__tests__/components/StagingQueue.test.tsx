import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { StagingQueue } from '@/components/StagingQueue';
import { BatchFileItem, BatchConfiguration } from '@/types/batch';

describe('StagingQueue Component', () => {
  const mockItems: BatchFileItem[] = [
    {
      id: 'file_1',
      file: new File(['mock1'], 'letter_18th_century.png', { type: 'image/png' }),
      name: 'letter_18th_century.png',
      size: 1024 * 500, // 500 KB
      type: 'image/png',
      status: 'staged',
      progress: 0,
      pageCount: 1,
    },
    {
      id: 'file_2',
      file: new File(['mock2'], 'legal_contract.pdf', { type: 'application/pdf' }),
      name: 'legal_contract.pdf',
      size: 1024 * 1024 * 2.5, // 2.5 MB
      type: 'application/pdf',
      status: 'staged',
      progress: 0,
      pageCount: 4,
    },
  ];

  const defaultConfig: BatchConfiguration = {
    outputMode: 'single',
    modelBias: 'general',
    format: 'markdown',
  };

  it('renders staging items with file names, page counts, and total payload size', () => {
    render(
      <StagingQueue generalOnly={false}
        items={mockItems}
        config={defaultConfig}
        onConfigChange={() => {}}
        onRemoveItem={() => {}}
        onAddMoreFiles={() => {}}
        onStartBatch={() => {}}
        onClearQueue={() => {}}
        isProcessing={false}
        overallProgress={0}
        currentProcessingIndex={0}
        overallStageText=""
      />
    );

    expect(screen.getByTestId('staging-queue-container')).toBeInTheDocument();
    expect(screen.getByText(/letter_18th_century.png/i)).toBeInTheDocument();
    expect(screen.getByText(/legal_contract.pdf/i)).toBeInTheDocument();
    expect(screen.getByText(/2 files/i)).toBeInTheDocument();
    expect(screen.getByText(/3.0 MB/i)).toBeInTheDocument(); // 0.5 + 2.5 = 3.0 MB
    expect(screen.getByTestId('btn-start-batch-execution')).toHaveTextContent(/Transcribe All/i);
  });

  it('allows changing batch configuration execution parameters', () => {
    const onConfigChange = vi.fn();
    render(
      <StagingQueue generalOnly={false}
        items={mockItems}
        config={defaultConfig}
        onConfigChange={onConfigChange}
        onRemoveItem={() => {}}
        onAddMoreFiles={() => {}}
        onStartBatch={() => {}}
        onClearQueue={() => {}}
        isProcessing={false}
        overallProgress={0}
        currentProcessingIndex={0}
        overallStageText=""
      />
    );

    // 1. Output mode toggle to discrete
    const discreteRadio = screen.getByTestId('config-output-discrete');
    fireEvent.click(discreteRadio);
    expect(onConfigChange).toHaveBeenCalledWith(
      expect.objectContaining({ outputMode: 'discrete' })
    );

    // 2. Model bias toggle to archival
    const archivalRadio = screen.getByTestId('config-bias-archival');
    fireEvent.click(archivalRadio);
    expect(onConfigChange).toHaveBeenCalledWith(
      expect.objectContaining({ modelBias: 'archival' })
    );

    // 3. Format toggle to json
    const jsonRadio = screen.getByTestId('config-format-json');
    fireEvent.click(jsonRadio);
    expect(onConfigChange).toHaveBeenCalledWith(
      expect.objectContaining({ format: 'json' })
    );
  });

  it('renders granular parallel progress meters when processing is active', () => {
    render(
      <StagingQueue generalOnly={false}
        items={mockItems}
        config={defaultConfig}
        onConfigChange={() => {}}
        onRemoveItem={() => {}}
        onAddMoreFiles={() => {}}
        onStartBatch={() => {}}
        onClearQueue={() => {}}
        isProcessing={true}
        overallProgress={65}
        currentProcessingIndex={1}
        overallStageText="Processing file 2 of 2 • Transcribing page 3..."
      />
    );

    expect(screen.getByTestId('batch-progress-meter')).toBeInTheDocument();
    expect(screen.getByTestId('batch-progress-header')).toHaveTextContent(/Processing file 2 of 2/i);
    expect(screen.getByTestId('batch-progress-stage')).toHaveTextContent(/Transcribing page 3/i);
    expect(screen.getByText('65%')).toBeInTheDocument();
  });

  it('fires remove and clear callbacks', () => {
    const onRemoveItem = vi.fn();
    const onClearQueue = vi.fn();

    render(
      <StagingQueue generalOnly={false}
        items={mockItems}
        config={defaultConfig}
        onConfigChange={() => {}}
        onRemoveItem={onRemoveItem}
        onAddMoreFiles={() => {}}
        onStartBatch={() => {}}
        onClearQueue={onClearQueue}
        isProcessing={false}
        overallProgress={0}
        currentProcessingIndex={0}
        overallStageText=""
      />
    );

    const removeBtn = screen.getByTestId('btn-remove-item-file_1');
    fireEvent.click(removeBtn);
    expect(onRemoveItem).toHaveBeenCalledWith('file_1');

    const clearBtn = screen.getByTestId('btn-clear-queue');
    fireEvent.click(clearBtn);
    expect(onClearQueue).toHaveBeenCalled();
  });
});
