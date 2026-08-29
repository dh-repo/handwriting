import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { SAMPLE_MESSY_CURSIVE } from '@/lib/sampleDocuments';

describe('InlineEditor Component', () => {
  it('renders line rows with confidence badges and inputs in structured mode', () => {
    render(<InlineEditor page={SAMPLE_MESSY_CURSIVE.pages[0]} />);

    expect(screen.getByTestId('inline-editor-container')).toBeInTheDocument();
    expect(screen.getByTestId('tab-structured')).toBeInTheDocument();
    expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();

    SAMPLE_MESSY_CURSIVE.pages[0].lines.forEach((line) => {
      expect(screen.getByTestId(`line-input-${line.line_id}`)).toBeInTheDocument();
    });
  });

  it('triggers onLineChange callback when typing in a line input', () => {
    const onLineChange = vi.fn();
    render(<InlineEditor page={SAMPLE_MESSY_CURSIVE.pages[0]} onLineChange={onLineChange} />);

    const firstInput = screen.getByTestId('line-input-p1_l1');
    fireEvent.change(firstInput, { target: { value: 'Corrected line text here' } });

    expect(onLineChange).toHaveBeenCalledWith('p1_l1', 'Corrected line text here');
  });

  it('switches to raw plaintext editor tab', () => {
    render(<InlineEditor page={SAMPLE_MESSY_CURSIVE.pages[0]} />);

    const rawTab = screen.getByTestId('tab-raw');
    fireEvent.click(rawTab);

    expect(screen.getByTestId('raw-textarea')).toBeInTheDocument();
    expect(screen.getByTestId('raw-textarea')).toHaveValue(SAMPLE_MESSY_CURSIVE.pages[0].full_text);
  });

  it('navigates to speed review tab and advances through uncertain words', () => {
    const onWordChange = vi.fn();
    render(
      <InlineEditor
        page={SAMPLE_MESSY_CURSIVE.pages[0]}
        onWordChange={onWordChange}
      />
    );

    const speedTab = screen.getByTestId('tab-speed-review');
    fireEvent.click(speedTab);

    expect(screen.getByTestId('speed-review-panel')).toBeInTheDocument();
    expect(screen.getByTestId('speed-review-input')).toBeInTheDocument();

    const speedInput = screen.getByTestId('speed-review-input');
    fireEvent.change(speedInput, { target: { value: 'migraines' } });

    const acceptBtn = screen.getByTestId('btn-speed-accept');
    fireEvent.click(acceptBtn);

    expect(onWordChange).toHaveBeenCalled();
  });

  it('triggers undo, redo, and revert callbacks when buttons are clicked', () => {
    const onUndo = vi.fn();
    const onRedo = vi.fn();
    const onRevertAll = vi.fn();

    const modifiedPage = {
      ...SAMPLE_MESSY_CURSIVE.pages[0],
      lines: SAMPLE_MESSY_CURSIVE.pages[0].lines.map((l, i) =>
        i === 0 ? { ...l, is_edited: true, text: 'Modified text' } : l
      ),
    };

    render(
      <InlineEditor
        page={modifiedPage}
        onUndo={onUndo}
        onRedo={onRedo}
        onRevertAll={onRevertAll}
        canUndo={true}
        canRedo={true}
      />
    );

    const undoBtn = screen.getByTestId('btn-undo');
    fireEvent.click(undoBtn);
    expect(onUndo).toHaveBeenCalled();

    const redoBtn = screen.getByTestId('btn-redo');
    fireEvent.click(redoBtn);
    expect(onRedo).toHaveBeenCalled();

    const revertBtn = screen.getByTestId('btn-revert-all');
    fireEvent.click(revertBtn);
    expect(onRevertAll).toHaveBeenCalled();
  });
});
