'use client';

import React, { useState, useRef, useCallback } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, AlertCircle, Loader2, Sparkles, Zap } from 'lucide-react';

export interface DropzoneProps {
  onFileAccepted: (file: File) => void;
  isLoading?: boolean;
  uploadProgress?: number;
  processingStage?: string;
  disabled?: boolean;
  maxSizeBytes?: number;
  className?: string;
  onSelectSample?: (sampleId: string) => void;
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
  processingStage = 'Processing document...',
  disabled = false,
  maxSizeBytes = 25 * 1024 * 1024,
  className = '',
  onSelectSample,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
      e.target.value = '';
    }
  };

  return (
    <div className={`w-full ${className}`}>
      <div
        data-testid="dropzone-container"
        onClick={handleClick}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
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
        className={`relative overflow-hidden flex flex-col items-center justify-center p-8 sm:p-12 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer select-none outline-none focus:ring-4 focus:ring-cyan-500/20 group ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-950/30 scale-[1.01] shadow-2xl shadow-cyan-500/20 ring-2 ring-cyan-400/40'
            : 'border-slate-800 hover:border-cyan-500/50 bg-slate-900/60 hover:bg-slate-900/80 shadow-xl backdrop-blur-xl'
        } ${disabled || isLoading ? 'pointer-events-none opacity-90' : ''}`}
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

        {/* Laser scanline animation during active processing */}
        {isLoading && <div className="animate-scanline" />}

        {isLoading ? (
          <div className="flex flex-col items-center text-center space-y-4 max-w-sm py-4">
            <div className="relative">
              <div className="w-16 h-16 rounded-full border-4 border-cyan-500/20 border-t-cyan-400 animate-spin" />
              <Loader2 className="w-7 h-7 text-cyan-400 absolute inset-0 m-auto animate-pulse" />
            </div>
            <div>
              <p className="font-bold text-white text-base tracking-tight flex items-center justify-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400 animate-pulse" />
                <span>Recognizing Handwriting...</span>
              </p>
              <p className="text-xs text-slate-400 mt-1 font-mono">{processingStage}</p>
            </div>
            {uploadProgress > 0 && (
              <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden mt-1 p-0.5 border border-slate-700/50">
                <div
                  className="bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-500 h-full rounded-full transition-all duration-300 ease-out shadow-sm shadow-cyan-500/50"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-500/20 via-indigo-500/20 to-purple-500/20 flex items-center justify-center text-cyan-400 border border-cyan-500/30 shadow-lg shadow-cyan-500/10 group-hover:scale-110 group-hover:border-cyan-400/60 transition-all duration-300">
              <UploadCloud className="w-8 h-8 transition-transform" />
            </div>

            <div className="space-y-1">
              <p className="text-base sm:text-lg font-bold text-white tracking-tight">
                Drag & drop your handwriting document here
              </p>
              <p className="text-xs sm:text-sm text-slate-400">
                or <span className="text-cyan-400 font-semibold hover:underline cursor-pointer">browse files</span> from your Mac / device
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-slate-800/80 text-slate-300 border border-slate-700/80 shadow-sm">
                <ImageIcon className="w-3.5 h-3.5 text-cyan-400" /> PNG, JPEG, TIFF
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-slate-800/80 text-slate-300 border border-slate-700/80 shadow-sm">
                <FileText className="w-3.5 h-3.5 text-indigo-400" /> Multi-page PDF
              </span>
              <span className="inline-flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-mono text-slate-400 bg-slate-800/40 border border-slate-700/40">
                Max 25 MB
              </span>
            </div>

            {onSelectSample && (
              <div
                className="pt-2 text-xs text-slate-400 flex items-center gap-2 flex-wrap justify-center"
                onClick={(e) => e.stopPropagation()}
              >
                <span className="text-slate-500">Quick Demo Presets:</span>
                <button
                  type="button"
                  onClick={() => onSelectSample('sample_clean_cursive')}
                  className="font-medium text-cyan-400 hover:text-cyan-300 px-2 py-0.5 rounded-md bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 transition-colors"
                >
                  Clean Cursive
                </button>
                <button
                  type="button"
                  onClick={() => onSelectSample('sample_legal_contract')}
                  className="font-medium text-indigo-400 hover:text-indigo-300 px-2 py-0.5 rounded-md bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/20 transition-colors"
                >
                  Legal & Signatures
                </button>
                <button
                  type="button"
                  onClick={() => onSelectSample('sample_prescription')}
                  className="font-medium text-purple-400 hover:text-purple-300 px-2 py-0.5 rounded-md bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/20 transition-colors"
                >
                  Doctor Prescription
                </button>
                <button
                  type="button"
                  onClick={() => onSelectSample('sample_multipage')}
                  className="font-medium text-emerald-400 hover:text-emerald-300 px-2 py-0.5 rounded-md bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 transition-colors"
                >
                  3-Page Multi-Document
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {errorMessage && (
        <div
          data-testid="dropzone-error"
          className="mt-3 p-3.5 rounded-2xl bg-rose-950/60 border border-rose-800 text-rose-300 text-xs shadow-lg shadow-rose-950/40 flex items-center gap-2.5 animate-in fade-in"
        >
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
