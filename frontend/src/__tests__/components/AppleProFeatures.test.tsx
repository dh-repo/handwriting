import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CommandPalette } from '../../components/CommandPalette';
import { MedicalEntitiesCard } from '../../components/MedicalEntitiesCard';
import { ShortcutsModal } from '../../components/ShortcutsModal';
import { DocumentViewer } from '../../components/DocumentViewer';
import { SplitCurtain } from '../../components/SplitCurtain';
import { SignatureInspector } from '../../components/SignatureInspector';
import { DarkroomToolbar } from '../../components/DarkroomToolbar';
import { DiagnosticsHUD } from '../../components/DiagnosticsHUD';
import { SAMPLE_PRESETS, SAMPLE_LEGAL_CONTRACT } from '../../lib/sampleDocuments';

describe('Apple Pro UX & SOTA Features Test Suite', () => {
  const sampleDoc = SAMPLE_PRESETS.sample_prescription;
  const samplePage = sampleDoc.pages[0];

  describe('1. CommandPalette (⌘K) Integration', () => {
    it('renders when open and lists commands', () => {
      const handleClose = vi.fn();
      render(
        <CommandPalette
          isOpen={true}
          onClose={handleClose}
          document={sampleDoc}
        />
      );

      expect(screen.getByTestId('command-palette-modal')).toBeInTheDocument();
      expect(screen.getByTestId('command-palette-input')).toBeInTheDocument();
      expect(screen.getByTestId('command-item-sample_clean_cursive')).toBeInTheDocument();
    });

    it('filters commands based on search input', async () => {
      const user = userEvent.setup();
      render(
        <CommandPalette
          isOpen={true}
          onClose={vi.fn()}
          document={sampleDoc}
        />
      );

      const input = screen.getByTestId('command-palette-input');
      await user.type(input, 'Prescription');

      expect(screen.getByTestId('command-item-sample_prescription')).toBeInTheDocument();
      expect(screen.queryByTestId('command-item-sample_clean_cursive')).not.toBeInTheDocument();
    });

    it('executes action and closes when command item is clicked', () => {
      const handleSelectSample = vi.fn();
      const handleClose = vi.fn();
      render(
        <CommandPalette
          isOpen={true}
          onClose={handleClose}
          document={sampleDoc}
          onSelectSample={handleSelectSample}
        />
      );

      const item = screen.getByTestId('command-item-sample_prescription');
      fireEvent.click(item);
      expect(handleSelectSample).toHaveBeenCalled();
      expect(handleClose).toHaveBeenCalled();
    });
  });

  describe('2. MedicalEntitiesCard Clinical Prescription Radar', () => {
    it('extracts and renders clinical prescription entities and sig translations', () => {
      render(<MedicalEntitiesCard page={samplePage} />);

      expect(screen.getByTestId('medical-entities-card')).toBeInTheDocument();
      expect(screen.getByText(/Clinical Prescription Radar/i)).toBeInTheDocument();
      expect(screen.getByText(/DEA\/NPI Valid/i)).toBeInTheDocument();
    });
  });

  describe('3. ShortcutsModal Sheet', () => {
    it('renders keyboard shortcuts list when open', () => {
      const handleClose = vi.fn();
      render(<ShortcutsModal isOpen={true} onClose={handleClose} />);

      expect(screen.getByTestId('shortcuts-modal-content')).toBeInTheDocument();
      expect(screen.getAllByText(/Keyboard Shortcuts/i)[0]).toBeInTheDocument();
      expect(screen.getByText(/⌘K \/ \//i)).toBeInTheDocument();
    });
  });

  describe('4. Retina Loupe 3x Magnifier in DocumentViewer', () => {
    it('renders loupe toggle button and handles active state', () => {
      const handleToggleLoupe = vi.fn();
      render(
        <DocumentViewer
          page={samplePage}
          isLoupeActive={true}
          onToggleLoupe={handleToggleLoupe}
        />
      );

      const loupeBtn = screen.getByTestId('btn-toggle-loupe');
      expect(loupeBtn).toBeInTheDocument();
      fireEvent.click(loupeBtn);
      expect(handleToggleLoupe).toHaveBeenCalled();
    });
  });

  describe('5. SplitCurtain X-Ray Comparison Slider', () => {
    it('renders curtain container and divider with clip-path overlay', () => {
      render(<SplitCurtain page={samplePage} />);

      expect(screen.getByTestId('split-curtain-container')).toBeInTheDocument();
      expect(screen.getByTestId('split-typeset-overlay')).toBeInTheDocument();
      expect(screen.getByTestId('split-curtain-divider')).toBeInTheDocument();
      expect(screen.getByText(/Original Scan/i)).toBeInTheDocument();
      expect(screen.getByText(/Digital Typeset/i)).toBeInTheDocument();
    });
  });

  describe('6. SignatureInspector & Legal Seal', () => {
    it('renders signature detection cards with entropy score and verification seal', () => {
      render(<SignatureInspector page={SAMPLE_LEGAL_CONTRACT.pages[0]} />);

      expect(screen.getByTestId('signature-inspector-container')).toBeInTheDocument();
      expect(screen.getByText(/Signature & Endorsement Verification/i)).toBeInTheDocument();
      expect(screen.getByText(/Verified Seal/i)).toBeInTheDocument();
    });
  });

  describe('7. DarkroomToolbar Image Adjustments', () => {
    it('renders presets and handles contrast / brightness changes', () => {
      const handleChange = vi.fn();
      render(
        <DarkroomToolbar
          settings={{ contrast: 1.0, brightness: 1.0, grayscale: false, inverted: false }}
          onChange={handleChange}
        />
      );

      expect(screen.getByTestId('darkroom-toolbar-container')).toBeInTheDocument();
      fireEvent.click(screen.getByTestId('btn-preset-faded'));
      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({ contrast: 2.2, brightness: 1.15 })
      );
    });
  });

  describe('8. DiagnosticsHUD Neural Telemetry', () => {
    it('renders latency breakdown and hardware status when open', () => {
      const handleClose = vi.fn();
      render(
        <DiagnosticsHUD
          document={sampleDoc}
          isOpen={true}
          onClose={handleClose}
        />
      );

      expect(screen.getByTestId('diagnostics-hud-content')).toBeInTheDocument();
      expect(screen.getByText(/Neural Pipeline Diagnostics HUD/i)).toBeInTheDocument();
      expect(screen.getByText(/Vision Encoder \(ViT-Large 304M\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Apple Silicon Metal Accelerator/i)).toBeInTheDocument();
    });
  });
});
