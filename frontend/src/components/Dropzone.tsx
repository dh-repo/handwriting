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
        className={`relative overflow-hidden flex flex-col items-center justify-center p-12 sm:p-16 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer select-none outline-none group ${
          isDragOver
            ? 'border-blue-400 bg-blue-500/15 scale-[1.01] shadow-2xl shadow-blue-500/30 ring-8 ring-blue-500/20'
            : 'border-white/20 hover:border-blue-400/60 bg-slate-900/50 hover:bg-slate-900/70 shadow-2xl backdrop-blur-2xl'
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
        <div className="absolute w-80 h-80 rounded-full bg-gradient-to-tr from-blue-600/20 to-purple-600/20 blur-3xl opacity-50 pointer-events-none group-hover:opacity-75 transition-opacity" />

        {isLoading ? (
          <div className="flex flex-col items-center text-center space-y-5 max-w-md py-4 relative z-10">
            {/* Apple Intelligence style pulsing ambient glow */}
            <div className="relative flex items-center justify-center">
              <div className="absolute w-24 h-24 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-500 blur-xl opacity-40 animate-pulse" />
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
          <div className="flex flex-col items-center text-center space-y-5 relative z-10 max-w-xl">
            {/* Apple HIG elevated icon button */}
            <div className="relative w-20 h-20 rounded-3xl bg-gradient-to-b from-white/10 to-white/5 flex items-center justify-center text-blue-400 border border-white/20 shadow-xl shadow-black/40 group-hover:scale-105 group-hover:border-blue-400/50 group-hover:shadow-blue-500/25 transition-all duration-300">
              <UploadCloud className="w-10 h-10 text-white/90 group-hover:text-blue-400 transition-colors" />
            </div>

            <div className="space-y-1.5">
              <p className="text-lg sm:text-2xl font-bold text-white tracking-tight">
                Drop your handwriting image or PDF here
              </p>
              <p className="text-xs sm:text-sm text-slate-300">
                or <span className="text-blue-400 font-semibold hover:text-blue-300 underline underline-offset-4 decoration-blue-400/40">drop files, folders, or browse</span>
              </p>
            </div>

            {/* Source Parity Badges: Clipboard, Camera/Scanner, Folders */}
            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <button
                type="button"
                data-testid="badge-clipboard-paste"
                onClick={handlePasteButtonClick}
                title="Paste image directly from clipboard"
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.06] hover:bg-white/[0.12] text-slate-200 hover:text-white border border-white/10 hover:border-blue-400/40 transition-all shadow-sm group/btn"
              >
                <ClipboardPaste className="w-3.5 h-3.5 text-indigo-400 group-hover/btn:scale-110 transition-transform" />
                <span>Paste from Clipboard (Cmd/Ctrl + V)</span>
              </button>

              <button
                type="button"
                data-testid="badge-camera-capture"
                onClick={handleCameraCaptureClick}
                title="Capture handwriting via device camera or document scanner"
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.06] hover:bg-white/[0.12] text-slate-200 hover:text-white border border-white/10 hover:border-blue-400/40 transition-all shadow-sm group/btn"
              >
                <Camera className="w-3.5 h-3.5 text-blue-400 group-hover/btn:scale-110 transition-transform" />
                <span>Camera / Scanner Capture</span>
              </button>

              <button
                type="button"
                data-testid="badge-browse-folder"
                onClick={handleBrowseFolderClick}
                title="Select an entire folder of handwriting scans"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.06] hover:bg-white/[0.12] text-slate-200 hover:text-white border border-white/10 hover:border-blue-400/40 transition-all shadow-sm group/btn"
              >
                <FolderOpen className="w-3.5 h-3.5 text-cyan-400 group-hover/btn:scale-110 transition-transform" />
                <span>Browse Folder</span>
              </button>
            </div>

            {/* Supported formats & size badge */}
            <div className="flex flex-wrap items-center justify-center gap-2 pt-2 border-t border-white/[0.07] w-full max-w-md">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-white/[0.04] text-slate-300 border border-white/10 shadow-sm">
                <ImageIcon className="w-3.5 h-3.5 text-blue-400" /> PNG, JPEG, TIFF
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-white/[0.04] text-slate-300 border border-white/10 shadow-sm">
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
