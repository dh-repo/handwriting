'use client';

import React, { useMemo } from 'react';
import { PenTool, Clock, Check, X } from 'lucide-react';
import { PageResult, SignatureDecision, SignatureReviewRecord } from '../types/ocr';
import {
  decisionFor,
  decisionLabel,
  findSignatureCandidates,
  kindLabel,
  pendingReviewCount,
} from '../lib/signatureCandidates';

export interface SignatureInspectorProps {
  page: PageResult;
  reviews?: SignatureReviewRecord[];
  selectedLineId?: string | null;
  onSelectLine?: (lineId: string) => void;
  onDecide?: (lineId: string, decision: SignatureDecision) => void;
  className?: string;
}

export const SignatureInspector: React.FC<SignatureInspectorProps> = ({
  page,
  reviews = [],
  selectedLineId = null,
  onSelectLine,
  onDecide,
  className = '',
}) => {
  const candidates = useMemo(() => findSignatureCandidates(page), [page]);
  const pending = pendingReviewCount(candidates, reviews, page.page_number);

  if (candidates.length === 0) return null;

  return (
    <div
      data-testid="signature-inspector-container"
      className={`rounded-2xl border border-slate-500/30 bg-slate-950/40 backdrop-blur-xl p-4 shadow-xl space-y-3 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-slate-500/20 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-xl bg-slate-500/20 text-slate-200 border border-slate-500/30 shadow-sm">
            <PenTool className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-white tracking-tight">
              Signature candidates
            </h4>
            <p className="text-[10px] text-slate-400">
              General-purpose sign-off lines. Human decision only. Not a match. Not medical.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1 text-[11px] font-mono text-amber-300 bg-amber-950/60 px-2 py-0.5 rounded-lg border border-amber-800/60 shadow-sm">
          <Clock className="w-3.5 h-3.5" />
          <span data-testid="signature-review-status">
            {pending > 0 ? `${pending} pending review` : 'Reviewed'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2">
        {candidates.map((sig) => {
          const decision = decisionFor(reviews, page.page_number, sig.lineId);
          const selected = selectedLineId === sig.lineId;
          return (
            <div
              key={sig.lineId}
              data-testid={`signature-card-${sig.lineId}`}
              data-decision={decision}
              className={`p-3 rounded-xl bg-slate-900/80 border transition-all flex flex-col justify-between space-y-2 group shadow-sm ${
                selected ? 'border-cyan-500/50' : 'border-slate-800/80 hover:border-slate-500/40'
              }`}
            >
              <button
                type="button"
                data-testid={`signature-select-${sig.lineId}`}
                onClick={() => onSelectLine?.(sig.lineId)}
                className="flex items-center justify-between gap-2 text-left"
              >
                <span className="text-xs font-medium text-white flex items-center gap-1.5 tracking-wide min-w-0">
                  <PenTool className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <span className="truncate">&ldquo;{sig.text}&rdquo;</span>
                </span>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-slate-500/10 text-slate-300 border border-slate-500/30 font-semibold shrink-0">
                  {kindLabel(sig.kind)}
                </span>
              </button>

              <div className="flex items-center justify-between gap-2 text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800">
                <span data-testid={`signature-decision-${sig.lineId}`}>
                  {decisionLabel(decision)}
                </span>
                <span className="text-slate-300 shrink-0">
                  {(sig.confidence * 100).toFixed(0)}% line conf
                </span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  data-testid={`signature-accept-${sig.lineId}`}
                  onClick={() => onDecide?.(sig.lineId, 'accepted')}
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium border ${
                    decision === 'accepted'
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      : 'bg-white/[0.04] text-slate-300 border-white/10 hover:bg-white/[0.08]'
                  }`}
                >
                  <Check className="w-3 h-3" />
                  Accept present
                </button>
                <button
                  type="button"
                  data-testid={`signature-reject-${sig.lineId}`}
                  onClick={() => onDecide?.(sig.lineId, 'rejected')}
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium border ${
                    decision === 'rejected'
                      ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                      : 'bg-white/[0.04] text-slate-300 border-white/10 hover:bg-white/[0.08]'
                  }`}
                >
                  <X className="w-3 h-3" />
                  Reject
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
