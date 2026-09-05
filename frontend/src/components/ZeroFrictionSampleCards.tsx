'use client';

import React from 'react';
import { Sparkles, ArrowRight, FileText, Scale, Stethoscope, CheckCircle2 } from 'lucide-react';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MESSY_CURSIVE } from '../lib/sampleDocuments';
import { DocumentOCRResult } from '../types/ocr';

export interface ZeroFrictionSampleCardsProps {
  onSelectSample: (sample: DocumentOCRResult) => void;
  className?: string;
}

export interface SampleCardItem {
  id: string;
  title: string;
  subtitle: string;
  category: string;
  accuracy: string;
  thumbnailUrl: string;
  icon: React.ReactNode;
  badgeStyle: string;
  glowStyle: string;
  data: DocumentOCRResult;
}

export const ZERO_FRICTION_SAMPLES: SampleCardItem[] = [
  {
    id: 'sample_18th_century',
    title: 'Cursive Note',
    subtitle: 'Example scan and prepared transcript for trying the review controls',
    category: 'Demo',
    accuracy: 'Prepared demo',
    thumbnailUrl: '/samples/sample_clean_cursive.png',
    icon: <FileText className="w-5 h-5 text-emerald-400" />,
    badgeStyle: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    glowStyle: 'hover:border-emerald-500/50 hover:shadow-emerald-500/20',
    data: SAMPLE_CLEAN_CURSIVE,
  },
  {
    id: 'sample_annotated_notes',
    title: 'Annotated Meeting Notes',
    subtitle: 'Prepared example for practicing corrections',
    category: 'Demo',
    accuracy: 'Prepared demo',
    thumbnailUrl: '/samples/sample_clean_cursive.png',
    icon: <Scale className="w-5 h-5 text-indigo-400" />,
    badgeStyle: 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30',
    glowStyle: 'hover:border-indigo-500/50 hover:shadow-indigo-500/20',
    data: SAMPLE_MESSY_CURSIVE,
  },

];

export const ZeroFrictionSampleCards: React.FC<ZeroFrictionSampleCardsProps> = ({
  onSelectSample,
  className = '',
}) => {
  return (
    <div data-testid="zero-friction-samples-container" className={`w-full max-w-4xl mx-auto space-y-3.5 ${className}`}>
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-blue-400" />
          <h3 className="text-xs sm:text-sm font-semibold text-zinc-300 tracking-tight">
            No document on hand? Try a labeled demo:
          </h3>
        </div>
        <span className="text-[11px] font-mono text-zinc-500 hidden sm:inline-flex">Prepared examples — not recognition benchmarks</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
        {ZERO_FRICTION_SAMPLES.map((sample) => (
          <button
            key={sample.id}
            type="button"
            data-testid={`sample-card-${sample.id}`}
            onClick={() => onSelectSample(structuredClone(sample.data))}
            className="group relative text-left p-4 rounded-2xl bg-zinc-900/50 hover:bg-zinc-800/50 border border-zinc-800 hover:border-zinc-700 transition-all duration-300 hover:scale-[1.015] shadow-xl backdrop-blur-xl flex flex-col justify-between overflow-hidden cursor-pointer"
          >
            {/* Ambient Corner Glow */}
            <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-white/[0.03] to-transparent rounded-bl-3xl pointer-events-none" />

            <div>
              {/* Header row with icon and accuracy badge */}
              <div className="flex items-center justify-between gap-2 mb-3">
                <div className="w-8 h-8 rounded-xl bg-zinc-800/90 border border-zinc-700/60 flex items-center justify-center group-hover:scale-105 transition-transform">
                  {sample.icon}
                </div>
                <span
                  className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2.5 py-0.5 rounded-full border shadow-sm ${sample.badgeStyle}`}
                >
                  <CheckCircle2 className="w-3 h-3" />
                  <span>{sample.accuracy}</span>
                </span>
              </div>

              {/* Title & Subtitle */}
              <h4 className="text-sm font-semibold text-zinc-100 group-hover:text-blue-300 transition-colors flex items-center justify-between mb-1">
                <span>{sample.title}</span>
                <ArrowRight className="w-3.5 h-3.5 text-blue-400 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
              </h4>

              <p className="text-xs text-zinc-400 leading-relaxed line-clamp-2 mb-3">
                {sample.subtitle}
              </p>
            </div>

            {/* Normalized 16:9 Thumbnail preview strip */}
            <div className="relative aspect-[16/9] w-full rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950/80 flex items-center justify-center mt-2 group-hover:border-zinc-700 transition-colors">
              <img
                src={sample.thumbnailUrl}
                alt={sample.title}
                className="w-full h-full object-cover object-center opacity-70 group-hover:opacity-90 group-hover:scale-105 transition-all duration-300"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-zinc-950/90 via-transparent to-transparent" />
              <span className="absolute bottom-2 left-2.5 text-[10px] font-mono text-zinc-300 font-medium">
                {sample.category}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
