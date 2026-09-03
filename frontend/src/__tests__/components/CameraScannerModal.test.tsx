import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CameraScannerModal } from '@/components/CameraScannerModal';

describe('CameraScannerModal Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.HTMLMediaElement.prototype.play = vi.fn().mockReturnValue(Promise.resolve());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders nothing when isOpen is false', () => {
    render(
      <CameraScannerModal
        isOpen={false}
        onClose={() => {}}
        onCapture={() => {}}
      />
    );
    expect(screen.queryByTestId('camera-scanner-modal')).not.toBeInTheDocument();
  });

  it('displays fallback state when camera is unavailable or permission is denied', async () => {
    // navigator.mediaDevices.getUserMedia rejected
    Object.defineProperty(global.navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockRejectedValue(new Error('Permission denied')),
      },
      writable: true,
      configurable: true,
    });

    const onFallback = vi.fn();
    const onClose = vi.fn();

    render(
      <CameraScannerModal
        isOpen={true}
        onClose={onClose}
        onCapture={() => {}}
        onFallbackToFilePicker={onFallback}
      />
    );

    expect(screen.getByTestId('camera-scanner-modal')).toBeInTheDocument();
    expect(await screen.findByText(/Camera Unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/Permission denied/i)).toBeInTheDocument();

    const fallbackBtn = screen.getByTestId('btn-scanner-fallback-file');
    fireEvent.click(fallbackBtn);
    expect(onClose).toHaveBeenCalled();
    expect(onFallback).toHaveBeenCalled();
  });

  it('renders live video stream and shutter button when camera stream is acquired', async () => {
    const mockTrack = { stop: vi.fn() };
    const mockStream = {
      getTracks: vi.fn().mockReturnValue([mockTrack]),
    };

    Object.defineProperty(global.navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValue(mockStream),
      },
      writable: true,
      configurable: true,
    });

    const onClose = vi.fn();
    render(
      <CameraScannerModal
        isOpen={true}
        onClose={onClose}
        onCapture={() => {}}
      />
    );

    expect(await screen.findByTestId('scanner-video-feed')).toBeInTheDocument();
    expect(screen.getByTestId('btn-scanner-shutter')).toBeInTheDocument();
    expect(screen.getByTestId('toggle-high-contrast')).toBeInTheDocument();

    // Toggle high-contrast filter
    const toggleContrast = screen.getByTestId('toggle-high-contrast');
    fireEvent.click(toggleContrast);

    // Close modal
    const closeBtn = screen.getByTestId('btn-close-scanner-modal');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });
});
