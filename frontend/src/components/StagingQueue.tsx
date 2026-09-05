'use client';

import React, { useRef } from 'react';
import {
  FileText,
  Trash2,
  Plus,
  Play,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Sparkles,
  Layers,
  Settings2,
  FileCode,
  FileType,
  FileSpreadsheet,
  FileCheck,
  Eye,
  SlidersHorizontal,
} from 'lucide-react';
import {
  BatchFileItem,
  BatchConfiguration,
  OutputMode,
  ModelBias,
  BatchExportFormat,
} from '../types/batch';
import { DocumentOCRResult } from '../types/ocr';

export interface StagingQueueProps {
  generalOnly?: boolean;
  items: BatchFileItem[];
  config: BatchConfiguration;
  onConfigChange: (newConfig: BatchConfiguration) => void;
  onRemoveItem: (id: string) => void;
  onAddMoreFiles: (files: File[]) => void;
  onStartBatch: () => void;
  onClearQueue: () => void;
  isProcessing: boolean;
  overallProgress: number;
  currentProcessingIndex: number;
  overallStageText: string;
  onSelectPreviewDoc?: (doc: DocumentOCRResult) => void;
  className?: string;
}

export const StagingQueue: React.FC<StagingQueueProps> = ({
  items,
  generalOnly = true,
  config,
  onConfigChange,
  onRemoveItem,
  onAddMoreFiles,
  onStartBatch,
  onClearQueue,
  isProcessing,
  overallProgress,
  currentProcessingIndex,
  overallStageText,
  onSelectPreviewDoc,
  className = '',
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const totalBytes = items.reduce((acc, it) => acc + it.size, 0);
  const totalMb = (totalBytes / (1024 * 1024)).toFixed(1);
  const completedCount = items.filter((it) => it.status === 'completed').length;

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onAddMoreFiles(Array.from(e.target.files));
      e.target.value = '';
    }
  };

  const handleAddFilesClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      data-testid="staging-queue-container"
      className={`w-full max-w-4xl mx-auto rounded-3xl border border-white/10 bg-slate-900/60 backdrop-blur-2xl p-6 sm:p-8 shadow-2xl space-y-6 ${className}`}
    >
      {/* Hidden Multi-File Input for "+ Add more" */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.pdf"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Header & Payload Summary */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-white/[0.08]">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" />
            <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight">
              Batch Ingestion Queue
            </h2>
            <span
              data-testid="queue-count-badge"
              className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/30"
            >
              {items.length} {items.length === 1 ? 'file' : 'files'}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Total payload: <span className="text-slate-200 font-mono font-medium">{totalMb} MB</span>
            {completedCount > 0 && (
              <span className="ml-2 text-emerald-400 font-medium">
                • {completedCount} of {items.length} complete
              </span>
            )}
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {!isProcessing && (
            <>
              <button
                type="button"
                data-testid="btn-add-more-files"
                onClick={handleAddFilesClick}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.06] hover:bg-white/[0.12] text-slate-200 hover:text-white border border-white/10 transition-colors shadow-sm"
              >
                <Plus className="w-3.5 h-3.5 text-blue-400" />
                <span>Add Files</span>
              </button>
              <button
                type="button"
                data-testid="btn-clear-queue"
                onClick={onClearQueue}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/20 transition-colors shadow-sm"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear All</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Granular Overall Progress Meter (When processing) */}
      {isProcessing && (
        <div
          data-testid="batch-progress-meter"
          className="p-4 rounded-2xl bg-blue-500/[0.07] border border-blue-500/20 space-y-2.5 animate-in fade-in"
        >
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 font-semibold text-white">
              <Sparkles className="w-4 h-4 text-blue-400 animate-pulse" />
              <span data-testid="batch-progress-header">
                Processing file {Math.min(items.length, currentProcessingIndex + 1)} of {items.length}
              </span>
            </div>
            <span className="font-mono text-blue-300 font-bold">{overallProgress}%</span>
          </div>

          <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden p-0.5 border border-white/5 shadow-inner">
            <div
              className="bg-gradient-to-r from-blue-500 via-indigo-400 to-purple-500 h-full rounded-full transition-all duration-300 ease-out"
              style={{ width: `${Math.max(5, overallProgress)}%` }}
            />
          </div>

          <p data-testid="batch-progress-stage" className="text-xs text-slate-400 font-normal truncate">
            {overallStageText || 'Executing vision-language transcription...'}
          </p>
        </div>
      )}

      {/* Staged Items List / Carousel */}
      <div
        data-testid="staging-items-list"
        className="space-y-2.5 max-h-[340px] overflow-y-auto pr-1 select-none"
      >
        {items.map((item, idx) => {
          const isItemActive = isProcessing && currentProcessingIndex === idx;
          const isItemDone = item.status === 'completed';
          const isItemError = item.status === 'error';
          const sizeKb = (item.size / 1024).toFixed(0);

          return (
            <div
              key={item.id}
              data-testid={`staged-item-${item.id}`}
              className={`flex items-center justify-between p-3 sm:p-3.5 rounded-2xl border transition-all duration-200 ${
                isItemActive
                  ? 'bg-blue-500/10 border-blue-500/40 ring-2 ring-blue-500/20'
                  : isItemDone
                  ? 'bg-emerald-950/20 border-emerald-500/30'
                  : isItemError
                  ? 'bg-rose-950/20 border-rose-500/30'
                  : 'bg-white/[0.03] hover:bg-white/[0.05] border-white/[0.07]'
              }`}
            >
              {/* Thumbnail & File Info */}
              <div className="flex items-center gap-3.5 min-w-0 flex-1">
                {/* Thumbnail */}
                <div className="w-12 h-12 rounded-xl bg-slate-950/80 border border-white/10 flex items-center justify-center overflow-hidden flex-shrink-0 relative">
                  {item.previewUrl ? (
                    <img
                      src={item.previewUrl}
                      alt={item.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = 'none';
                      }}
                    />
                  ) : (
                    <FileText className="w-6 h-6 text-slate-400" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/70 text-[9px] text-center font-mono uppercase text-slate-300 truncate px-0.5">
                    {item.name.split('.').pop() || 'FILE'}
                  </span>
                </div>

                {/* Meta details */}
                <div className="min-w-0 flex-1 space-y-0.5">
                  <p className="text-sm font-semibold text-white truncate" title={item.name}>
                    {item.name}
                  </p>
                  <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                    <span>{sizeKb} KB</span>
                    <span>•</span>
                    <span>{item.pageCount || 1} {item.pageCount === 1 ? 'page' : 'pages'}</span>
                    {item.stageText && (
                      <>
                        <span>•</span>
                        <span className="text-blue-300 truncate max-w-[200px]">{item.stageText}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Status / Actions */}
              <div className="flex items-center gap-3 flex-shrink-0 ml-3">
                {item.status === 'staged' && !isProcessing && (
                  <button
                    type="button"
                    data-testid={`btn-remove-item-${item.id}`}
                    onClick={() => onRemoveItem(item.id)}
                    title="Remove from batch"
                    className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}

                {item.status === 'queued' && (
                  <span className="text-xs text-slate-500 font-mono px-2.5 py-1 rounded-lg bg-white/5 border border-white/5">
                    Queued
                  </span>
                )}

                {item.status === 'processing' && (
                  <div className="flex items-center gap-2 text-xs text-blue-400 font-semibold px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Transcribing...</span>
                  </div>
                )}

                {item.status === 'completed' && (
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-medium px-2 py-0.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>Ready</span>
                    </span>
                    {item.result && onSelectPreviewDoc && (
                      <button
                        type="button"
                        data-testid={`btn-review-item-${item.id}`}
                        onClick={() => onSelectPreviewDoc(item.result!)}
                        className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-white shadow-sm transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Review</span>
                      </button>
                    )}
                  </div>
                )}

                {item.status === 'error' && (
                  <span className="inline-flex items-center gap-1 text-xs text-rose-400 font-medium px-2 py-0.5 rounded-lg bg-rose-500/10 border border-rose-500/20">
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span>Failed</span>
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Batch Configuration Section */}
      <div
        data-testid="batch-configuration-panel"
        className="p-4 sm:p-5 rounded-2xl bg-white/[0.02] border border-white/[0.08] space-y-4"
      >
        <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
          <Settings2 className="w-4 h-4 text-blue-400" />
          <span>Batch Execution Configuration</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          {/* 1. Output Mode */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-300 block">Output Mode</label>
            <div className="space-y-1">
              <label
                data-testid="config-output-single"
                className={`flex items-center gap-2 p-2 rounded-xl border cursor-pointer transition-colors ${
                  config.outputMode === 'single'
                    ? 'bg-blue-500/15 border-blue-500/40 text-white font-medium'
                    : 'bg-white/[0.02] border-white/5 text-slate-400 hover:text-white'
                }`}
              >
                <input
                  type="radio"
                  name="outputMode"
                  checked={config.outputMode === 'single'}
                  onChange={() => onConfigChange({ ...config, outputMode: 'single' })}
                  disabled={isProcessing}
                  className="hidden"
                />
                <span className="w-2 h-2 rounded-full bg-current" />
                <span>Combine into single export</span>
              </label>

              <label
                data-testid="config-output-discrete"
                className={`flex items-center gap-2 p-2 rounded-xl border cursor-pointer transition-colors ${
                  config.outputMode === 'discrete'
                    ? 'bg-blue-500/15 border-blue-500/40 text-white font-medium'
                    : 'bg-white/[0.02] border-white/5 text-slate-400 hover:text-white'
                }`}
              >
                <input
                  type="radio"
                  name="outputMode"
                  checked={config.outputMode === 'discrete'}
                  onChange={() => onConfigChange({ ...config, outputMode: 'discrete' })}
                  disabled={isProcessing}
                  className="hidden"
                />
                <span className="w-2 h-2 rounded-full bg-current" />
                <span>Discrete files per document</span>
              </label>
            </div>
          </div>

          {/* 2. Model Bias */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-300 block">Model Bias</label>
            <div className="space-y-1">
              {(
                [
                  { id: 'general', label: 'General Cursive', desc: 'Balanced baseline' },
                  { id: 'archival', label: 'Legal / Archival', desc: 'Signatures & historical' },
                  { id: 'tabular', label: 'Tabular / Forms', desc: 'Structured alignment' },
                ] as const
              ).filter(bias => !generalOnly || bias.id === "general").map((bias) => (
                <label
                  key={bias.id}
                  data-testid={`config-bias-${bias.id}`}
                  className={`flex items-center gap-2 p-2 rounded-xl border cursor-pointer transition-colors ${
                    config.modelBias === bias.id
                      ? 'bg-indigo-500/15 border-indigo-500/40 text-white font-medium'
                      : 'bg-white/[0.02] border-white/5 text-slate-400 hover:text-white'
                  }`}
                >
                  <input
                    type="radio"
                    name="modelBias"
                    checked={config.modelBias === bias.id}
                    onChange={() => onConfigChange({ ...config, modelBias: bias.id })}
                    disabled={isProcessing}
                    className="hidden"
                  />
                  <span className="w-2 h-2 rounded-full bg-current" />
                  <div className="flex items-center justify-between w-full">
                    <span>{bias.label}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* 3. Export Format */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-300 block">Export Format</label>
            <div className="space-y-1">
              {(
                [
                  { id: 'markdown', label: 'Markdown (.md)' },
                  { id: 'text', label: 'Plain Text (.txt)' },
                  { id: 'json', label: 'JSON with BBoxes' },
                  { id: 'pdf', label: 'Searchable PDF' },
                ] as const
              ).map((fmt) => (
                <label
                  key={fmt.id}
                  data-testid={`config-format-${fmt.id}`}
                  className={`flex items-center gap-2 p-2 rounded-xl border cursor-pointer transition-colors ${
                    config.format === fmt.id
                      ? 'bg-purple-500/15 border-purple-500/40 text-white font-medium'
                      : 'bg-white/[0.02] border-white/5 text-slate-400 hover:text-white'
                  }`}
                >
                  <input
                    type="radio"
                    name="format"
                    checked={config.format === fmt.id}
                    onChange={() => onConfigChange({ ...config, format: fmt.id })}
                    disabled={isProcessing}
                    className="hidden"
                  />
                  <span className="w-2 h-2 rounded-full bg-current" />
                  <span>{fmt.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Start Batch Execution Footer */}
      <div className="pt-2">
        <button
          type="button"
          data-testid="btn-start-batch-execution"
          onClick={onStartBatch}
          disabled={isProcessing || items.length === 0}
          className="w-full py-4 rounded-2xl font-bold text-sm sm:text-base text-white bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-500 hover:via-indigo-500 hover:to-purple-500 shadow-xl shadow-blue-500/25 border border-white/20 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-40 disabled:pointer-events-none hover:scale-[1.008]"
        >
          {isProcessing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Transcribing Batch Queue ({overallProgress}%)...</span>
            </>
          ) : (
            <>
              <Play className="w-5 h-5 fill-current" />
              <span>Transcribe All ({items.length} {items.length === 1 ? 'Document' : 'Documents'})</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
