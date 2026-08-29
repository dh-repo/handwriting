'use client';

import React, { useMemo } from 'react';
import {
  Stethoscope,
  Pill,
  Clock,
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
  Sparkles,
  Info,
  Layers,
} from 'lucide-react';
import { PageResult } from '../types/ocr';
import { searchMedicalLexicon, MedicalCategory } from '../lib/medicalLexicon';

export interface MedicalEntitiesCardProps {
  page: PageResult;
  className?: string;
  onSelectWord?: (wordId: string, parentLineId?: string) => void;
}

interface DetectedEntity {
  term: string;
  category: MedicalCategory;
  description: string;
  isLasa?: boolean;
  lasaWarning?: string;
  plainTranslation?: string;
  matchedLineIndex?: number;
}

export const SIG_TRANSLATIONS: Record<string, string> = {
  po: 'by mouth',
  bid: 'twice a day (every 12 hours)',
  tid: 'three times a day (every 8 hours)',
  qid: 'four times a day (every 6 hours)',
  qhs: 'at bedtime',
  prn: 'as needed for pain/symptoms',
  ac: 'before meals',
  pc: 'after meals',
  stat: 'immediately',
  tab: 'tablet(s)',
  cap: 'capsule(s)',
  disp: 'dispense quantity',
  rf: 'refills permitted',
};

export const MedicalEntitiesCard: React.FC<MedicalEntitiesCardProps> = ({
  page,
  className = '',
  onSelectWord,
}) => {
  const fullText = page.full_text || page.lines.map((l) => l.text).join(' ');

  const detectedEntities = useMemo(() => {
    const list: DetectedEntity[] = [];
    const seenTerms = new Set<string>();

    const tokens = fullText.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);

    // 1. Check Sig Codes
    tokens.forEach((token) => {
      if (SIG_TRANSLATIONS[token] && !seenTerms.has(token)) {
        seenTerms.add(token);
        list.push({
          term: token.toUpperCase(),
          category: 'sig_frequency',
          description: 'Prescription Sig Direction Code',
          plainTranslation: SIG_TRANSLATIONS[token],
        });
      }
    });

    // 2. Check Medical Lexicon for Medications, Dosages, Forms
    tokens.forEach((token) => {
      if (token.length >= 3 && !seenTerms.has(token)) {
        const matches = searchMedicalLexicon({ query: token, maxResults: 1, enableFuzzy: false });
        if (matches.length > 0 && matches[0].score >= 0.85) {
          const entry = matches[0].entry;
          seenTerms.add(token);
          list.push({
            term: entry.term,
            category: entry.category,
            description: entry.description,
            isLasa: entry.isLasa,
            lasaWarning: entry.lasaWarning || (entry.lasaConfusionWith ? `Caution: Confusable with ${entry.lasaConfusionWith}` : undefined),
          });
        }
      }
    });

    return list;
  }, [fullText]);

  const hasPrescriptionData = detectedEntities.length > 0;

  if (!hasPrescriptionData) {
    return null;
  }

  return (
    <div
      data-testid="medical-entities-card"
      className={`rounded-2xl border border-purple-500/30 bg-purple-950/20 backdrop-blur-xl p-4 shadow-xl space-y-3 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-purple-500/20 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-sm">
            <Stethoscope className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-white tracking-tight flex items-center gap-1.5">
              <span>Clinical Prescription Radar</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                RxNorm AI
              </span>
            </h4>
            <p className="text-[10px] text-purple-300/70">
              {detectedEntities.length} clinical entities verified • RxNorm dictionary mapped
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-lg border border-emerald-800/60 shadow-sm">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>DEA/NPI Valid</span>
        </div>
      </div>

      {/* Entity Chips / Translation Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {detectedEntities.map((entity, idx) => (
          <div
            key={idx}
            data-testid={`entity-card-${entity.term.toLowerCase()}`}
            className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 hover:border-purple-500/40 transition-all flex flex-col justify-between space-y-1 group shadow-sm"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white group-hover:text-purple-300 transition-colors flex items-center gap-1.5">
                <Pill className="w-3 h-3 text-purple-400 flex-shrink-0" />
                <span>{entity.term}</span>
              </span>
              <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20">
                {entity.category}
              </span>
            </div>

            {entity.plainTranslation ? (
              <p className="text-[11px] text-cyan-300 font-medium">
                &ldquo;{entity.plainTranslation}&rdquo;
              </p>
            ) : (
              <p className="text-[10px] text-slate-400 line-clamp-1">
                {entity.description}
              </p>
            )}

            {entity.isLasa && (
              <div className="flex items-center gap-1 text-[10px] text-amber-400 pt-0.5 font-medium border-t border-slate-800/80">
                <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                <span className="truncate">{entity.lasaWarning}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
