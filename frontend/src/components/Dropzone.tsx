'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  UploadCloud,
  FileText,
  Image as ImageIcon,
  AlertCircle,
  Loader2,
  Sparkles,
  ClipboardPaste,
  Camera,
  FolderOpen,
  CheckCircle2,
} from 'lucide-react';
import { DocumentOCRResult } from '../types/ocr';
import { ZeroFrictionSampleCards } from './ZeroFrictionSampleCards';
import { CameraScannerModal } from './CameraScannerModal';

export interface DropzoneProps {
  onFileAccepted: (file: File) => void;
  onFilesAccepted?: (files: File[]) => void;
  onSelectSample?: (sample: DocumentOCRResult) => void;
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
  onFilesAccepted,
  onSelectSample,
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
  const folderInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [isCameraModalOpen, setIsCameraModalOpen] = useState(false);

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

  const validateFiles = useCallback(
    (files: File[]): File[] => {
      setErrorMessage(null);
      const validFiles: File[] = [];

      for (const file of files) {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        const isValidType =
          ACCEPTED_MIME_TYPES.includes(file.type) || ACCEPTED_EXTENSIONS.includes(ext);

        if (!isValidType) {
          setErrorMessage(
            `Unsupported file format "${file.name}". Please upload a PNG, JPEG, TIFF, or PDF document.`
          );
          continue;
        }

        if (file.size > maxSizeBytes) {
          const sizeMb = (maxSizeBytes / (1024 * 1024)).toFixed(0);
          setErrorMessage(
            `File "${file.name}" exceeds the ${sizeMb} MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`
          );
          continue;
        }

        validFiles.push(file);
      }

      return validFiles;
    },
    [maxSizeBytes]
  );

  const handleFilesDispatched = useCallback(
    (files: File[]) => {
      const validFiles = validateFiles(files);
      if (validFiles.length === 0) return;

      if (validFiles.length > 1 && onFilesAccepted) {
        onFilesAccepted(validFiles);
      } else {
        if (onFilesAccepted) {
          onFilesAccepted(validFiles);
        }
        onFileAccepted(validFiles[0]);
      }
    },
    [validateFiles, onFilesAccepted, onFileAccepted]
  );

  // Global Clipboard Paste Listener (Cmd/Ctrl + V)
  useEffect(() => {
    if (disabled || isLoading) return;

    const handleWindowPaste = (e: ClipboardEvent) => {
      const clipboardData = e.clipboardData;
      if (!clipboardData) return;

      const pastedFiles: File[] = [];
      const items = clipboardData.items;

      if (items && items.length > 0) {
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          if (item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            if (blob) {
              const pastedFile = new File(
                [blob],
                `clipboard_capture_${Date.now()}.png`,
                { type: blob.type || 'image/png' }
              );
              pastedFiles.push(pastedFile);
            }
          }
        }
      }

      if (pastedFiles.length > 0) {
        e.preventDefault();
        handleFilesDispatched(pastedFiles);
      }
    };

    window.addEventListener('paste', handleWindowPaste);
    return () => {
      window.removeEventListener('paste', handleWindowPaste);
    };
  }, [disabled, isLoading, handleFilesDispatched]);

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

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (disabled || isLoading) return;

    const droppedFiles: File[] = [];

    // Directory traversal if webkit entries available
    const items = e.dataTransfer.items;
    if (items && items.length > 0 && typeof items[0].webkitGetAsEntry === 'function') {
      const promises: Promise<void>[] = [];

      const readEntry = async (entry: any) => {
        if (!entry) return;
        if (entry.isFile) {
          await new Promise<void>((resolve) => {
            entry.file((f: File) => {
              droppedFiles.push(f);
              resolve();
            }, () => resolve());
          });
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          const readBatch = async (): Promise<void> => {
            const entries: any[] = await new Promise((resolve) => {
              reader.readEntries((r: any[]) => resolve(r || []), () => resolve([]));
            });
            if (entries.length > 0) {
              for (const sub of entries) {
                await readEntry(sub);
              }
              await readBatch();
            }
          };
          await readBatch();
        }
      };

      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry();
        if (entry) {
          promises.push(readEntry(entry));
        }
      }

      await Promise.all(promises);
    }

    if (droppedFiles.length === 0 && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        droppedFiles.push(e.dataTransfer.files[i]);
      }
    }

    if (droppedFiles.length > 0) {
      handleFilesDispatched(droppedFiles);
    }
  };

  const handleClick = () => {
    if (!disabled && !isLoading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesDispatched(Array.from(e.target.files));
      e.target.value = '';
    }
  };

  const handleFolderInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesDispatched(Array.from(e.target.files));
      e.target.value = '';
    }
  };

  const handleCameraInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesDispatched(Array.from(e.target.files));
      e.target.value = '';
    }
  };

  const handlePasteButtonClick = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (disabled || isLoading) return;

    try {
      if (navigator.clipboard && typeof navigator.clipboard.read === 'function') {
        const clipboardItems = await navigator.clipboard.read();
        const files: File[] = [];

        for (const item of clipboardItems) {
          const imageType = item.types.find((t) => t.startsWith('image/'));
          if (imageType) {
            const blob = await item.getType(imageType);
            files.push(
              new File([blob], `clipboard_${Date.now()}.png`, { type: imageType })
            );
          }
        }

        if (files.length > 0) {
          handleFilesDispatched(files);
          return;
        }
      }
    } catch {
      // Browser permission prompt or fallback
    }

    setErrorMessage('Please press Cmd+V (or Ctrl+V) to paste an image from your clipboard.');
  };

  const handleCameraCaptureClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (disabled || isLoading) return;
    setIsCameraModalOpen(true);
  };

  const handleBrowseFolderClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (disabled || isLoading) return;
    folderInputRef.current?.click();
  };

  // Determine active stage index for multi-step progress
  const getCurrentStepIndex = (progress: number, stageText: string): number => {
    const lower = stageText.toLowerCase();
    if (progress >= 95 || lower.includes('complete')) return 3;
    if (
      lower.includes('decod') ||
      lower.includes('analyz') ||
      lower.includes('extract') ||
      lower.includes('stroke') ||
      lower.includes('transformer') ||
      progress >= 45
    ) {
      return 2;
    }
    if (lower.includes('segment') || progress >= 25) return 1;
    return 0;
  };

  const currentStep = getCurrentStepIndex(uploadProgress, processingStage);

  const PIPELINE_STEPS = [
    {
      id: 'ingestion',
      title: 'Document Ingestion',
      detail: 'Deskew & contrast normalization',
    },
    {
      id: 'segmentation',
      title: 'Stroke & Layout',
      detail: 'Line crops & baseline detection',
    },
    {
      id: 'decoding',
      title: 'Neural VLM Decoding',
      detail: 'Transformer attention over cursive',
    },
    {
      id: 'verification',
      title: 'Calibration & Verification',
      detail: 'Confidence scoring & entities',
    },
  ];

  return (
    <div className={`w-full max-w-4xl mx-auto space-y-6 ${className}`}>
      {/* Expanded Main Dropzone Container */}
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
        className={`relative overflow-hidden flex flex-col items-center justify-center p-8 sm:p-12 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer select-none outline-none group ${
          isDragOver
            ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01] shadow-2xl shadow-indigo-500/20 ring-4 ring-indigo-500/20'
            : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/40 hover:bg-zinc-900/60 shadow-2xl backdrop-blur-2xl'
        } ${disabled || isLoading ? 'pointer-events-none' : ''}`}
      >
        {/* Hidden inputs */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          data-testid="dropzone-input"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={handleFileInputChange}
          className="hidden"
          disabled={disabled || isLoading}
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          // @ts-expect-error webkitdirectory attribute
          webkitdirectory=""
          directory=""
          onChange={handleFolderInputChange}
          className="hidden"
          disabled={disabled || isLoading}
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleCameraInputChange}
          className="hidden"
          disabled={disabled || isLoading}
        />

        {/* Ambient Gradient Glow */}
        <div className="absolute w-80 h-80 rounded-full bg-gradient-to-tr from-indigo-600/15 via-blue-600/10 to-transparent blur-3xl opacity-40 pointer-events-none group-hover:opacity-65 transition-opacity" />

        {isLoading ? (
          /* Multi-Step Neural Processing Status Cockpit */
          <div className="w-full max-w-2xl py-2 px-2 sm:px-4 space-y-6 relative z-10 text-left">
            {/* Header: Status badge & Elapsed timer */}
            <div className="flex items-center justify-between gap-3 border-b border-zinc-800/80 pb-4">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-mono font-medium tracking-wide text-blue-400 bg-blue-500/10 border border-blue-500/25">
                <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                <span>NEURAL RECOGNITION ACTIVE</span>
              </div>
              <div className="inline-flex items-center gap-1.5 text-[11px] font-mono text-zinc-400 bg-zinc-800/80 border border-zinc-700/60 px-3 py-1 rounded-full">
                <span>⏱</span>
                <span>{elapsedSeconds.toFixed(1)}s elapsed</span>
              </div>
            </div>

            {/* Title & Stage message */}
            <div className="space-y-1.5 text-center sm:text-left">
              <h3 className="font-bold text-zinc-100 text-lg sm:text-xl tracking-tight flex items-center justify-center sm:justify-start gap-2.5">
                <Sparkles className="w-5 h-5 text-blue-400 animate-pulse flex-shrink-0" />
                <span>Transcribing Handwriting...</span>
              </h3>
              <p className="text-xs sm:text-sm text-zinc-400 font-normal tracking-wide">
                {processingStage}
              </p>
            </div>

            {/* Accessible Progress Meter */}
            <div
              role="progressbar"
              aria-valuenow={uploadProgress}
              aria-valuemin={0}
              aria-valuemax={100}
              className="space-y-1.5"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-zinc-400">Recognition Pipeline</span>
                <span className="text-blue-400 font-bold">{Math.max(10, Math.round(uploadProgress))}%</span>
              </div>
              <div className="w-full bg-zinc-800/90 rounded-full h-2 overflow-hidden p-0.5 border border-zinc-700/60 shadow-inner">
                <div
                  className="bg-gradient-to-r from-blue-500 via-indigo-400 to-emerald-400 h-full rounded-full transition-all duration-500 ease-out shadow-md shadow-blue-500/20 relative"
                  style={{ width: `${Math.max(10, Math.min(100, uploadProgress))}%` }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse" />
                </div>
              </div>
            </div>

            {/* Stepped Pipeline Radar (4 Stages) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 pt-1">
              {PIPELINE_STEPS.map((step, idx) => {
                const isCompleted = idx < currentStep;
                const isActive = idx === currentStep;
                const isPending = idx > currentStep;

                return (
                  <div
                    key={step.id}
                    className={`p-3 rounded-xl border transition-all duration-300 backdrop-blur-md flex flex-col justify-between ${
                      isCompleted
                        ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300 shadow-sm'
                        : isActive
                        ? 'bg-blue-500/10 border-blue-500/40 text-blue-200 ring-2 ring-blue-500/20 shadow-md shadow-blue-500/10'
                        : 'bg-zinc-900/40 border-zinc-800/70 text-zinc-500'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1.5 mb-1.5">
                      <span className="text-[10px] font-mono font-medium tracking-wider uppercase">
                        Step {idx + 1}
                      </span>
                      {isCompleted ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-400 font-semibold">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Done</span>
                        </span>
                      ) : isActive ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-mono text-blue-400 font-semibold animate-pulse">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          <span>Active</span>
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono text-zinc-600">Pending</span>
                      )}
                    </div>
                    <div>
                      <p
                        className={`text-xs font-semibold tracking-tight ${
                          isCompleted
                            ? 'text-zinc-200'
                            : isActive
                            ? 'text-white'
                            : 'text-zinc-500'
                        }`}
                      >
                        {step.title}
                      </p>
                      <p className="text-[10px] text-zinc-400 mt-0.5 leading-snug line-clamp-2">
                        {step.detail}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Live Activity Log Pill */}
            <div className="bg-zinc-950/90 rounded-xl border border-zinc-800/80 px-3.5 py-2.5 flex items-center justify-between font-mono text-xs text-zinc-300 shadow-inner">
              <div className="flex items-center gap-2.5 truncate">
                <span className="text-blue-400 select-none flex-shrink-0 font-bold">›</span>
                <span className="text-zinc-500">neural-engine:</span>
                <span className="text-zinc-300 truncate">active stream • stroke decoder online</span>
              </div>
              <span className="w-1.5 h-3.5 bg-blue-400 animate-pulse select-none flex-shrink-0" />
            </div>
          </div>
        ) : (
          /* ReadMe-Inspired Idle Capsule Dropzone */
          <div className="flex flex-col items-center text-center space-y-4 relative z-10 max-w-xl">
            {/* Elevated Icon Container */}
            <div className="w-14 h-14 rounded-2xl bg-zinc-800/80 border border-zinc-700/60 flex items-center justify-center text-zinc-300 shadow-lg group-hover:scale-105 group-hover:border-blue-400/50 group-hover:text-blue-400 transition-all duration-300">
              <UploadCloud className="w-7 h-7 transition-colors" />
            </div>

            <div className="space-y-1">
              <h2 className="text-xl sm:text-2xl font-bold text-zinc-100 tracking-tight">
                Drop your handwriting image or PDF here
              </h2>
              <p className="text-xs sm:text-sm text-zinc-400 max-w-md mx-auto">
                Transcribe general handwriting, then review and correct the result
              </p>
            </div>

            {/* Primary Action Button */}
            <div className="pt-1">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleClick();
                }}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full text-xs sm:text-sm font-semibold bg-zinc-100 hover:bg-white text-zinc-950 shadow-md hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
              >
                <span>Browse Files</span>
              </button>
            </div>

            {/* ReadMe-Style Inline Helper Pill Row */}
            <div className="flex flex-wrap items-center justify-center gap-2 pt-2 text-xs text-zinc-400">
              <span className="text-zinc-500 font-medium">Try it out —</span>
              <button
                type="button"
                data-testid="badge-clipboard-paste"
                onClick={handlePasteButtonClick}
                title="Paste image directly from clipboard (Cmd+V)"
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-800/60 hover:bg-zinc-800 text-zinc-300 hover:text-zinc-100 border border-zinc-700/50 transition-all cursor-pointer shadow-sm"
              >
                <ClipboardPaste className="w-3 h-3 text-zinc-400" />
                <span>Paste Clipboard</span>
              </button>
              <button
                type="button"
                data-testid="badge-camera-capture"
                onClick={handleCameraCaptureClick}
                title="Capture handwriting via device camera or document scanner"
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-800/60 hover:bg-zinc-800 text-zinc-300 hover:text-zinc-100 border border-zinc-700/50 transition-all cursor-pointer shadow-sm"
              >
                <Camera className="w-3 h-3 text-zinc-400" />
                <span>Camera / Scanner</span>
              </button>
              <button
                type="button"
                data-testid="badge-browse-folder"
                onClick={handleBrowseFolderClick}
                title="Select an entire folder of handwriting scans"
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-800/60 hover:bg-zinc-800 text-zinc-300 hover:text-zinc-100 border border-zinc-700/50 transition-all cursor-pointer shadow-sm"
              >
                <FolderOpen className="w-3 h-3 text-zinc-400" />
                <span>Browse Folder</span>
              </button>
            </div>

            {/* Formats & Limit footnote */}
            <div className="flex flex-wrap items-center justify-center gap-2 pt-2 text-[11px] text-zinc-500">
              <span className="inline-flex items-center gap-1">
                <ImageIcon className="w-3 h-3 text-zinc-400" /> PNG, JPEG, TIFF
              </span>
              <span className="text-zinc-700 select-none">•</span>
              <span className="inline-flex items-center gap-1">
                <FileText className="w-3 h-3 text-zinc-400" /> Multi-page PDF
              </span>
              <span className="text-zinc-700 select-none">•</span>
              <span className="font-mono text-zinc-400">Up to 50 MB</span>
            </div>
          </div>
        )}
      </div>

      {errorMessage && (
        <div
          data-testid="dropzone-error"
          className="p-3.5 rounded-2xl bg-rose-950/50 border border-rose-500/30 text-rose-300 text-xs shadow-xl flex items-center gap-2.5 backdrop-blur-xl animate-in fade-in"
        >
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Zero-Friction Sample Cards: 1-Click previews */}
      {!isLoading && onSelectSample && (
        <ZeroFrictionSampleCards onSelectSample={onSelectSample} />
      )}

      {/* Live Camera / Document Scanner Viewfinder Modal */}
      <CameraScannerModal
        isOpen={isCameraModalOpen}
        onClose={() => setIsCameraModalOpen(false)}
        onCapture={(file) => {
          handleFilesDispatched([file]);
        }}
        onFallbackToFilePicker={() => {
          cameraInputRef.current?.click();
        }}
      />
    </div>
  );
};
