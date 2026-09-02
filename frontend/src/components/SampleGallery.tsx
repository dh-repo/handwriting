'use client';

import React from 'react';
import { FileText, Stethoscope, Layers, FileEdit, Scale, ArrowRight, Sparkles } from 'lucide-react';
import { SAMPLE_PRESETS, SAMPLE_LEGAL_CONTRACT } from '../lib/sampleDocuments';
import { DocumentOCRResult } from '../types/ocr';

export interface SampleGalleryProps {
  onSelectSample: (sample: DocumentOCRResult) => void;
  className?: string;
}

export const SAMPLE_CARD_CONFIGS = [
  {
    id: 'sample_clean_cursive',
    title: 'Cursive Correspondence',
    description: 'Crisp Spencerian & cursive script with smooth connected baseline flow.',
    category: 'Cursive',
    difficulty: 'Clean Script',
    icon: <FileText className="w-5 h-5 text-emerald-400" />,
    badgeColor: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    glowColor: 'group-hover:border-emerald-500/50 group-hover:shadow-emerald-500/10',
  },
  {
    id: 'sample_legal_contract',
    title: 'Legal Contract & Signature',
    description: 'Agreement clauses with dates and a general-purpose sign-off line for human review.',
    category: 'Legal / Signatures',
    difficulty: 'Signatures',
    icon: <Scale className="w-5 h-5 text-indigo-400" />,
    badgeColor: 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30',
    glowColor: 'group-hover:border-indigo-500/50 group-hover:shadow-indigo-500/10',
  },
  {
    id: 'sample_prescription',
    title: 'Doctor Prescription Slip',
    description: 'Clinic header with medication lines (Amoxicillin, Ibuprofen) and dosage sigs.',
    category: 'Medical / Clinical',
    difficulty: 'Prescription',
    icon: <Stethoscope className="w-5 h-5 text-purple-400" />,
    badgeColor: 'bg-purple-500/10 text-purple-300 border-purple-500/30',
    glowColor: 'group-hover:border-purple-500/50 group-hover:shadow-purple-500/10',
  },
  {
    id: 'sample_messy_cursive',
    title: 'Messy Journal & Field Note',
    description: 'Rapid unconstrained cursive notes with lower-confidence word review.',
    category: 'Notes / Forms',
    difficulty: 'Messy Cursive',
    icon: <FileEdit className="w-5 h-5 text-amber-400" />,
    badgeColor: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    glowColor: 'group-hover:border-amber-500/50 group-hover:shadow-amber-500/10',
  },
  {
    id: 'sample_multipage',
    title: '3-Page Multi-Document Record',
    description: 'Comprehensive record: intake assessment, orders, and archival sign-off.',
    category: 'Multi-Page',
    difficulty: 'Multi-Page PDF',
    icon: <Layers className="w-5 h-5 text-cyan-400" />,
    badgeColor: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
    glowColor: 'group-hover:border-cyan-500/50 group-hover:shadow-cyan-500/10',
  },
];

export const SampleGallery: React.FC<SampleGalleryProps> = ({ onSelectSample, className = '' }) => {
  return (
    <div data-testid="sample-gallery-container" className={`space-y-3.5 ${className}`}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span>Universal Handwriting Sample Gallery</span>
        </h3>
        <span className="text-xs text-slate-400 font-mono">1-Click Live Previews</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {SAMPLE_CARD_CONFIGS.map((config) => {
          const presetData = SAMPLE_PRESETS[config.id] || (config.id === 'sample_legal_contract' ? SAMPLE_LEGAL_CONTRACT : null);
          if (!presetData) return null;

          return (
            <button
              key={config.id}
              data-testid={`sample-card-${config.id}`}
              onClick={() => onSelectSample(structuredClone(presetData))}
              className={`flex flex-col text-left p-4 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 hover:scale-[1.02] shadow-lg transition-all duration-300 group ${config.glowColor}`}
            >
              <div className="flex items-center justify-between w-full mb-3">
                <div className="p-2.5 rounded-xl bg-slate-800/80 border border-slate-700/60 group-hover:scale-110 transition-transform duration-200">
                  {config.icon}
                </div>
                <span className={`text-[10px] uppercase font-mono font-semibold px-2.5 py-0.5 rounded-full border shadow-sm ${config.badgeColor}`}>
                  {config.difficulty}
                </span>
              </div>

              <h4 className="font-bold text-white text-sm mb-1.5 group-hover:text-cyan-300 transition-colors flex items-center justify-between">
                <span className="truncate">{config.title}</span>
                <ArrowRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all text-cyan-400 flex-shrink-0" />
              </h4>

              <p className="text-xs text-slate-400 line-clamp-2 mb-3.5 flex-1 leading-relaxed">
                {config.description}
              </p>

              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-2.5 border-t border-slate-800/80 w-full">
                <span className="text-slate-400">{presetData.total_pages} {presetData.total_pages === 1 ? 'page' : 'pages'}</span>
                <span className="text-emerald-400 font-semibold flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  {((presetData.mean_confidence ?? presetData.overall_confidence ?? 0.95) * 100).toFixed(0)}% acc
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
