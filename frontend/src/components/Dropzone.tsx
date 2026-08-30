'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, AlertCircle, Loader2, Sparkles } from 'lucide-react';

export interface DropzoneProps {
  onFileAccepted: (file: File) => void;
  isLoading?: boolean;
  uploadProgress?: number;
  processingStage?: string;
  disabled?: boolean;
  maxSizeBytes?: number;
  className?: string;
}

const ACCEPTED_MIME_TYPES = [
  'image/png',
  'image/jpeg',
  'image/jpg',
  'image/tiff',
  'image/bmp',
  'image/webp',
  'application/pdf',
];

const ACCEPTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp', '.pdf'];

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileAccepted,
  isLoading = false,
  uploadProgress = 0,
  processingStage = 'Analyzing handwriting strokes...',
  disabled = false,
  maxSizeBytes = 50 * 1024 * 1024,
  className = '',
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Timer for active processing
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isLoading) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => +(prev + 0.5).toFixed(1));
      }, 500);
    } else {
      setElapsedSeconds(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isLoading]);

  const validateAndProcessFile = useCallback(
    (file: File) => {
      setErrorMessage(null);
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      const isValidType =
        ACCEPTED_MIME_TYPES.includes(file.type) || ACCEPTED_EXTENSIONS.includes(ext);

      if (!isValidType) {
        setErrorMessage(
          `Unsupported file format "${file.name}". Please upload a PNG, JPEG, TIFF, or PDF document.`
        );
        return;
      }

      if (file.size > maxSizeBytes) {
        const sizeMb = (maxSizeBytes / (1024 * 1024)).toFixed(0);
        setErrorMessage(
          `File exceeds the ${sizeMb} MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`
        );
        return;
      }

      onFileAccepted(file);
    },
    [maxSizeBytes, onFileAccepted]
  );

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled && !isLoading) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (disabled || isLoading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    if (!disabled && !isLoading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  return (
    <div className={`w-full max-w-2xl mx-auto ${className}`}>
      <div
        data-testid="dropzone-container"
        onClick={handleClick}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload handwriting image or PDF"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleClick();
          }
        }}
        className={`relative overflow-hidden flex flex-col items-center justify-center p-10 sm:p-14 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer select-none outline-none group ${
          isDragOver
            ? 'border-blue-400 bg-blue-500/10 scale-[1.01] shadow-2xl shadow-blue-500/20 ring-4 ring-blue-500/20'
            : 'border-white/15 hover:border-blue-400/50 bg-slate-900/40 hover:bg-slate-900/60 shadow-2xl backdrop-blur-2xl'
        } ${disabled || isLoading ? 'pointer-events-none' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          data-testid="dropzone-input"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={handleFileInputChange}
          className="hidden"
          disabled={disabled || isLoading}
        />

        {isLoading ? (
          <div className="flex flex-col items-center text-center space-y-5 max-w-md py-4">
            {/* Apple Intelligence style pulsing ambient glow */}
            <div className="relative flex items-center justify-center">
              <div className="absolute w-20 h-20 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-500 blur-xl opacity-40 animate-pulse" />
              <div className="w-16 h-16 rounded-2xl bg-white/[0.08] border border-white/20 backdrop-blur-md flex items-center justify-center shadow-inner">
                <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="font-semibold text-white text-base sm:text-lg tracking-tight flex items-center justify-center gap-2">
                <Sparkles className="w-4 h-4 text-blue-400 animate-pulse" />
                <span>Transcribing Handwriting...</span>
              </p>
              <p className="text-xs text-slate-400 font-normal tracking-wide">{processingStage}</p>
              {elapsedSeconds > 0 && (
                <p className="text-[11px] text-slate-500 font-mono pt-1">
                  Processing time: {elapsedSeconds.toFixed(1)}s
                </p>
              )}
            </div>

            {/* Smooth progress indicator */}
            <div className="w-64 sm:w-80 bg-white/10 rounded-full h-1.5 overflow-hidden p-0.5 border border-white/5 shadow-inner">
              <div
                className="bg-gradient-to-r from-blue-500 via-indigo-400 to-purple-500 h-full rounded-full transition-all duration-500 ease-out shadow-sm"
                style={{ width: `${Math.max(15, uploadProgress)}%` }}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-4">
            {/* Apple HIG elevated icon button */}
            <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-b from-white/10 to-white/5 flex items-center justify-center text-blue-400 border border-white/20 shadow-xl shadow-black/40 group-hover:scale-105 group-hover:border-blue-400/50 group-hover:shadow-blue-500/20 transition-all duration-300">
              <UploadCloud className="w-8 h-8 text-white/90 group-hover:text-blue-400 transition-colors" />
            </div>

            <div className="space-y-1">
              <p className="text-base sm:text-lg font-semibold text-white tracking-tight">
                Drop your handwriting image or PDF here
              </p>
              <p className="text-xs sm:text-sm text-slate-400">
                or <span className="text-blue-400 font-medium hover:text-blue-300 transition-colors">choose a file</span> from your device
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-white/[0.06] text-slate-300 border border-white/10 shadow-sm">
                <ImageIcon className="w-3.5 h-3.5 text-blue-400" /> PNG, JPEG, TIFF
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-white/[0.06] text-slate-300 border border-white/10 shadow-sm">
                <FileText className="w-3.5 h-3.5 text-indigo-400" /> Multi-page PDF
              </span>
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono text-slate-400 bg-white/[0.03] border border-white/5">
                Up to 50 MB
              </span>
            </div>
          </div>
        )}
      </div>

      {errorMessage && (
        <div
          data-testid="dropzone-error"
          className="mt-4 p-3.5 rounded-2xl bg-rose-950/50 border border-rose-500/30 text-rose-300 text-xs shadow-xl flex items-center gap-2.5 backdrop-blur-xl animate-in fade-in"
        >
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
