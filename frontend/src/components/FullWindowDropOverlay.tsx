'use client';

import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, Sparkles, FolderOpen, FileText, Image as ImageIcon } from 'lucide-react';

export interface FullWindowDropOverlayProps {
  onFilesDropped: (files: File[]) => void;
  disabled?: boolean;
}

export const FullWindowDropOverlay: React.FC<FullWindowDropOverlayProps> = ({
  onFilesDropped,
  disabled = false,
}) => {
  const [isWindowDragging, setIsWindowDragging] = useState(false);
  const dragCounterRef = useRef(0);

  useEffect(() => {
    if (disabled) return;

    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current += 1;

      // Only show overlay if dragging files
      if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
        setIsWindowDragging(true);
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'copy';
      }
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
      if (dragCounterRef.current === 0) {
        setIsWindowDragging(false);
      }
    };

    const handleDrop = async (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current = 0;
      setIsWindowDragging(false);

      if (!e.dataTransfer) return;

      const collectedFiles: File[] = [];

      // Support webkit directory traversal if available
      const items = e.dataTransfer.items;
      if (items && items.length > 0 && typeof items[0].webkitGetAsEntry === 'function') {
        const entryPromises: Promise<void>[] = [];

        const readEntry = async (entry: any) => {
          if (!entry) return;
          if (entry.isFile) {
            await new Promise<void>((resolve) => {
              entry.file(
                (file: File) => {
                  collectedFiles.push(file);
                  resolve();
                },
                () => resolve()
              );
            });
          } else if (entry.isDirectory) {
            const reader = entry.createReader();
            const readBatch = async (): Promise<void> => {
              const entries: any[] = await new Promise((resolve) => {
                reader.readEntries((results: any[]) => resolve(results || []), () => resolve([]));
              });
              if (entries.length > 0) {
                for (const subEntry of entries) {
                  await readEntry(subEntry);
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
            entryPromises.push(readEntry(entry));
          }
        }

        await Promise.all(entryPromises);
      }

      // Fallback or augment with standard dataTransfer.files
      if (collectedFiles.length === 0 && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        for (let i = 0; i < e.dataTransfer.files.length; i++) {
          collectedFiles.push(e.dataTransfer.files[i]);
        }
      }

      if (collectedFiles.length > 0) {
        onFilesDropped(collectedFiles);
      }
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);

    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
    };
  }, [disabled, onFilesDropped]);

  if (!isWindowDragging) return null;

  return (
    <div
      data-testid="full-window-drop-overlay"
      className="fixed inset-0 z-50 pointer-events-auto flex items-center justify-center p-6 sm:p-12 bg-[#07090e]/90 backdrop-blur-2xl transition-all duration-300 animate-in fade-in"
    >
      <div className="relative w-full h-full max-w-5xl max-h-[85vh] rounded-3xl border-4 border-dashed border-blue-400 bg-blue-500/[0.07] ring-8 ring-blue-500/20 shadow-[0_0_100px_rgba(59,130,246,0.35)] flex flex-col items-center justify-center text-center p-8 sm:p-14 select-none animate-pulse">
        {/* Ambient Glowing Aura */}
        <div className="absolute w-72 h-72 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-500 blur-3xl opacity-30 pointer-events-none" />

        {/* Floating Elevated Icon */}
        <div className="relative w-24 h-24 sm:w-28 sm:h-28 rounded-3xl bg-gradient-to-tr from-blue-600 to-indigo-500 border border-white/30 flex items-center justify-center shadow-2xl shadow-blue-500/50 mb-6 scale-110">
          <UploadCloud className="w-12 h-12 sm:w-14 sm:h-14 text-white animate-bounce" />
        </div>

        {/* Assertive Typography */}
        <div className="space-y-3 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider text-blue-300 bg-blue-500/20 border border-blue-400/40">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span>Active Drop Zone</span>
          </div>

          <h2 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight">
            Drop Files or Folders Anywhere
          </h2>

          <p className="text-base sm:text-lg text-slate-300 font-normal leading-relaxed">
            Release to ingest documents into the batch queue for high-throughput neural transcription.
          </p>
        </div>

        {/* Format Badges */}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-8">
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold bg-white/10 text-white border border-white/20 shadow-lg backdrop-blur-md">
            <ImageIcon className="w-4 h-4 text-blue-400" /> PNG, JPEG, TIFF
          </span>
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold bg-white/10 text-white border border-white/20 shadow-lg backdrop-blur-md">
            <FileText className="w-4 h-4 text-indigo-400" /> Multi-page PDFs
          </span>
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold bg-white/10 text-white border border-white/20 shadow-lg backdrop-blur-md">
            <FolderOpen className="w-4 h-4 text-cyan-400" /> Nested Folders
          </span>
        </div>
      </div>
    </div>
  );
};
