'use client';

import React from 'react';
import { Gauge, Type, ListOrdered, AlertTriangle, Clock, Zap, CheckCircle2 } from 'lucide-react';
import { DocumentOCRResult } from '../types/ocr';
import { formatDurationMs } from '../lib/utils';
import { getConfidenceColor } from '../lib/colorUtils';

export interface MetricsSummaryProps {
  document: DocumentOCRResult;
  onStartSpeedReview?: () => void;
  className?: string;
}

export const MetricsSummary: React.FC<MetricsSummaryProps> = ({
  document,
  onStartSpeedReview,
  className = '',
}) => {
  let totalWords = 0;
  let totalLines = 0;
  let lowConfWords = 0;

  document.pages.forEach((page) => {
    totalLines += page.lines.length;
    page.lines.forEach((line) => {
      if (line.words && line.words.length > 0) {
        totalWords += line.words.length;
        line.words.forEach((w) => {
          if (w.confidence < 0.70) lowConfWords++;
        });
      } else {
        totalWords += line.text.trim().split(/\s+/).filter(Boolean).length;
        if (line.confidence < 0.70) lowConfWords++;
      }
    });
  });

  const meanConfidence =
    document.mean_confidence ??
    document.overall_confidence ??
    (document.pages.reduce((acc, p) => acc + p.mean_confidence, 0) / (document.pages.length || 1));

  const confidenceStyle = getConfidenceColor(meanConfidence);

  return (
    <div
      data-testid="metrics-summary-container"
      className={`grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-2.5 ${className}`}
    >
      {/* Mean Confidence KPI */}
      <div className="flex items-center gap-3 p-3 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 shadow-md hover:border-slate-700 transition-colors">
        <div className={`p-2.5 rounded-xl border shadow-sm ${confidenceStyle.badgeBg} ${confidenceStyle.badgeBorder}`}>
          <Gauge className={`w-4 h-4 ${confidenceStyle.tailwindText}`} />
        </div>
        <div>
          <p className="text-[11px] font-medium text-slate-400">Mean Confidence</p>
          <p
            data-testid="metric-mean-confidence"
            className={`text-sm font-bold font-mono tracking-tight ${confidenceStyle.tailwindText}`}
          >
            {(meanConfidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Total Words KPI */}
      <div className="flex items-center gap-3 p-3 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 shadow-md hover:border-slate-700 transition-colors">
        <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shadow-sm">
          <Type className="w-4 h-4" />
        </div>
        <div>
          <p className="text-[11px] font-medium text-slate-400">Total Words</p>
          <p data-testid="metric-total-words" className="text-sm font-bold font-mono text-white tracking-tight">
            {totalWords}
          </p>
        </div>
      </div>

      {/* Total Lines KPI */}
      <div className="flex items-center gap-3 p-3 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 shadow-md hover:border-slate-700 transition-colors">
        <div className="p-2.5 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400 shadow-sm">
          <ListOrdered className="w-4 h-4" />
        </div>
        <div>
          <p className="text-[11px] font-medium text-slate-400">Lines Extracted</p>
          <p data-testid="metric-total-lines" className="text-sm font-bold font-mono text-white tracking-tight">
            {totalLines}
          </p>
        </div>
      </div>

      {/* Uncertain Tokens KPI */}
      <div className="flex items-center gap-3 p-3 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 shadow-md hover:border-slate-700 transition-colors">
        <div
          className={`p-2.5 rounded-xl border shadow-sm ${
            lowConfWords > 0
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
          }`}
        >
          {lowConfWords > 0 ? (
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-medium text-slate-400">Uncertain Words</p>
          <div className="flex items-center justify-between">
            <p
              data-testid="metric-low-conf-count"
              className={`text-sm font-bold font-mono tracking-tight ${
                lowConfWords > 0 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              {lowConfWords}
            </p>
            {lowConfWords > 0 && onStartSpeedReview && (
              <button
                type="button"
                data-testid="btn-metric-speed-review"
                onClick={onStartSpeedReview}
                className="text-[10px] font-bold text-cyan-400 hover:text-cyan-300 hover:underline bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/20"
              >
                Review
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Latency KPI */}
      <div className="col-span-2 sm:col-span-4 lg:col-span-1 flex items-center gap-3 p-3 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 shadow-md hover:border-slate-700 transition-colors">
        <div className="p-2.5 rounded-xl bg-slate-800 border border-slate-700/80 text-slate-300 shadow-sm">
          <Clock className="w-4 h-4 text-cyan-400" />
        </div>
        <div>
          <p className="text-[11px] font-medium text-slate-400">Inference Latency</p>
          <p data-testid="metric-latency" className="text-sm font-bold font-mono text-white tracking-tight">
            {formatDurationMs(document.processing_time_ms)}
          </p>
        </div>
      </div>
    </div>
  );
};
