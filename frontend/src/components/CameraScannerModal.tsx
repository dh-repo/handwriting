'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Camera,
  X,
  RefreshCw,
  Check,
  Sliders,
  AlertCircle,
  UploadCloud,
  Sparkles,
} from 'lucide-react';

export interface CameraScannerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
  onFallbackToFilePicker?: () => void;
}

export const CameraScannerModal: React.FC<CameraScannerModalProps> = ({
  isOpen,
  onClose,
  onCapture,
  onFallbackToFilePicker,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [hasCamera, setHasCamera] = useState<boolean | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [capturedBlobUrl, setCapturedBlobUrl] = useState<string | null>(null);
  const [capturedFile, setCapturedFile] = useState<File | null>(null);
  const [highContrastMode, setHighContrastMode] = useState<boolean>(true);

  const stopStream = useCallback(() => {
    if (streamRef.current && typeof streamRef.current.getTracks === 'function') {
      try {
        const tracks = streamRef.current.getTracks();
        if (Array.isArray(tracks)) {
          tracks.forEach((track) => {
            try {
              track.stop();
            } catch {
              // ignore
            }
          });
        }
      } catch {
        // ignore
      }
      streamRef.current = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    stopStream();
    setErrorMessage(null);
    setCapturedBlobUrl(null);
    setCapturedFile(null);

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setHasCamera(false);
      setErrorMessage('Camera access is not supported by your current browser environment.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        try {
          videoRef.current.srcObject = stream;
          const playPromise = videoRef.current.play();
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {});
          }
        } catch {
          // ignore jsdom play() unimplemented
        }
      }
      setHasCamera(true);
    } catch (err: unknown) {
      setHasCamera(false);
      const msg =
        err instanceof Error
          ? err.message
          : 'Unable to access camera. Please check browser permissions.';
      setErrorMessage(msg);
    }
  }, [stopStream]);

  useEffect(() => {
    if (isOpen) {
      startCamera();
    } else {
      stopStream();
      if (capturedBlobUrl) {
        URL.revokeObjectURL(capturedBlobUrl);
      }
    }
    return () => {
      stopStream();
    };
  }, [isOpen, startCamera, stopStream, capturedBlobUrl]);

  const handleCaptureSnapshot = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (highContrastMode) {
      ctx.filter = 'contrast(1.4) grayscale(0.8)';
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File(
        [blob],
        `camera_scan_${new Date().toISOString().slice(0, 10)}_${Date.now().toString().slice(-4)}.png`,
        { type: 'image/png' }
      );
      const url = URL.createObjectURL(blob);
      setCapturedFile(file);
      setCapturedBlobUrl(url);
    }, 'image/png', 0.95);
  };

  const handleConfirmScan = () => {
    if (capturedFile) {
      onCapture(capturedFile);
      onClose();
    }
  };

  const handleRetake = () => {
    if (capturedBlobUrl) {
      URL.revokeObjectURL(capturedBlobUrl);
    }
    setCapturedBlobUrl(null);
    setCapturedFile(null);
    if (videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  };

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      data-testid="camera-scanner-modal"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-xl animate-in fade-in duration-200"
    >
      <div className="relative w-full max-w-2xl rounded-3xl bg-[#0B0F19] border border-white/10 shadow-2xl overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-blue-500/20 text-blue-400 flex items-center justify-center border border-blue-500/30">
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white tracking-tight">
                Live Document Scanner
              </h3>
              <p className="text-[11px] text-slate-400">
                Align handwritten page within the illuminated guidelines
              </p>
            </div>
          </div>

          <button
            type="button"
            data-testid="btn-close-scanner-modal"
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Viewfinder / Capture Area */}
        <div className="relative aspect-[4/3] bg-black flex items-center justify-center overflow-hidden">
          {hasCamera === false || errorMessage ? (
            <div className="flex flex-col items-center justify-center p-6 text-center space-y-4 max-w-sm">
              <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center">
                <AlertCircle className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm font-semibold text-white">Camera Unavailable</p>
                <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                  {errorMessage || 'Camera access was blocked or is not supported.'}
                </p>
              </div>
              {onFallbackToFilePicker && (
                <button
                  type="button"
                  data-testid="btn-scanner-fallback-file"
                  onClick={() => {
                    onClose();
                    onFallbackToFilePicker();
                  }}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20 transition-all"
                >
                  <UploadCloud className="w-4 h-4" />
                  <span>Choose File from Device</span>
                </button>
              )}
            </div>
          ) : capturedBlobUrl ? (
            /* Frozen Snapshot Preview */
            <div className="relative w-full h-full flex items-center justify-center bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={capturedBlobUrl}
                alt="Captured scan"
                data-testid="captured-scan-preview"
                className="max-w-full max-h-full object-contain"
              />
              <div className="absolute top-3 left-3 px-3 py-1 rounded-full text-[11px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5 backdrop-blur-md">
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span>Page Snapshot Ready</span>
              </div>
            </div>
          ) : (
            /* Live Camera Feed */
            <div className="relative w-full h-full flex items-center justify-center">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                data-testid="scanner-video-feed"
                className={`w-full h-full object-cover transition-all ${
                  highContrastMode ? 'contrast-125 grayscale' : ''
                }`}
              />

              {/* Viewfinder Framing Overlay */}
              <div className="absolute inset-8 sm:inset-12 border-2 border-white/30 rounded-2xl pointer-events-none flex flex-col justify-between p-2">
                <div className="flex justify-between">
                  <div className="w-6 h-6 border-t-2 border-l-2 border-cyan-400 rounded-tl-lg" />
                  <div className="w-6 h-6 border-t-2 border-r-2 border-cyan-400 rounded-tr-lg" />
                </div>
                <div className="self-center px-3 py-1 rounded-full bg-black/60 text-[10px] text-cyan-300 font-mono tracking-wider uppercase border border-cyan-500/30 backdrop-blur-md">
                  Target Handwriting Area
                </div>
                <div className="flex justify-between">
                  <div className="w-6 h-6 border-b-2 border-l-2 border-cyan-400 rounded-bl-lg" />
                  <div className="w-6 h-6 border-b-2 border-r-2 border-cyan-400 rounded-br-lg" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Controls Bar */}
        <div className="flex items-center justify-between px-6 py-4 bg-white/[0.02] border-t border-white/10">
          <div className="flex items-center gap-2">
            {!capturedBlobUrl && hasCamera && (
              <button
                type="button"
                data-testid="toggle-high-contrast"
                onClick={() => setHighContrastMode((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-colors ${
                  highContrastMode
                    ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
                    : 'bg-white/[0.06] text-slate-300 border-white/10 hover:text-white'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>B&W Document Filter</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            {capturedBlobUrl ? (
              <>
                <button
                  type="button"
                  data-testid="btn-scanner-retake"
                  onClick={handleRetake}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5 text-slate-300" />
                  <span>Retake</span>
                </button>
                <button
                  type="button"
                  data-testid="btn-scanner-confirm"
                  onClick={handleConfirmScan}
                  className="inline-flex items-center gap-1.5 px-5 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/20 border border-white/20 transition-all"
                >
                  <Check className="w-3.5 h-3.5" />
                  <span>Use This Scan</span>
                </button>
              </>
            ) : hasCamera ? (
              <button
                type="button"
                data-testid="btn-scanner-shutter"
                onClick={handleCaptureSnapshot}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/25 border border-white/20 transition-all"
              >
                <div className="w-2 h-2 rounded-full bg-white animate-ping" />
                <span>Capture Page</span>
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};
