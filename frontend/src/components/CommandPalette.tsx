'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Search,
  FileText,
  Stethoscope,
  Scale,
  Layers,
  FileEdit,
  Download,
  Copy,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCw,
  Eye,
  FastForward,
  Sparkles,
  Sliders,
  Keyboard,
  X,
} from 'lucide-react';
import { DocumentOCRResult } from '../types/ocr';
import { SAMPLE_PRESETS, SAMPLE_LEGAL_CONTRACT } from '../lib/sampleDocuments';

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  document: DocumentOCRResult | null;
  onSelectSample?: (sample: DocumentOCRResult) => void;
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onZoomFit?: () => void;
  onRotate?: () => void;
  onToggleHeatmap?: () => void;
  onToggleBBoxes?: () => void;
  onToggleLoupe?: () => void;
  onStartSpeedReview?: () => void;
  onExportTxt?: () => void;
  onExportJson?: () => void;
  onExportCsv?: () => void;
  onCopyText?: () => void;
  onToggleInvert?: () => void;
}

interface CommandItem {
  id: string;
  title: string;
  category: 'Presets' | 'View & Canvas' | 'Analysis & Review' | 'Export & Share';
  shortcut?: string;
  icon: React.ReactNode;
  action: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  document,
  onSelectSample,
  onZoomIn,
  onZoomOut,
  onZoomFit,
  onRotate,
  onToggleHeatmap,
  onToggleBBoxes,
  onToggleLoupe,
  onStartSpeedReview,
  onExportTxt,
  onExportJson,
  onExportCsv,
  onCopyText,
  onToggleInvert,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const commands: CommandItem[] = useMemo(() => {
    const list: CommandItem[] = [];

    // Presets
    list.push({
      id: 'sample_clean_cursive',
      title: 'Load Clean Cursive Correspondence',
      category: 'Presets',
      shortcut: '⌘1',
      icon: <FileText className="w-4 h-4 text-emerald-400" />,
      action: () => {
        const p = SAMPLE_PRESETS.sample_clean_cursive;
        if (p) onSelectSample?.(structuredClone(p));
      },
    });
    list.push({
      id: 'sample_prescription',
      title: 'Load Doctor Prescription Slip',
      category: 'Presets',
      shortcut: '⌘2',
      icon: <Stethoscope className="w-4 h-4 text-purple-400" />,
      action: () => {
        const p = SAMPLE_PRESETS.sample_prescription;
        if (p) onSelectSample?.(structuredClone(p));
      },
    });
    list.push({
      id: 'sample_legal_contract',
      title: 'Load Legal Contract & Signatures',
      category: 'Presets',
      shortcut: '⌘3',
      icon: <Scale className="w-4 h-4 text-indigo-400" />,
      action: () => {
        onSelectSample?.(structuredClone(SAMPLE_LEGAL_CONTRACT));
      },
    });
    list.push({
      id: 'sample_messy_cursive',
      title: 'Load Messy Journal & Field Note',
      category: 'Presets',
      shortcut: '⌘4',
      icon: <FileEdit className="w-4 h-4 text-amber-400" />,
      action: () => {
        const p = SAMPLE_PRESETS.sample_messy_cursive;
        if (p) onSelectSample?.(structuredClone(p));
      },
    });
    list.push({
      id: 'sample_multipage',
      title: 'Load 3-Page Multi-Document PDF',
      category: 'Presets',
      shortcut: '⌘5',
      icon: <Layers className="w-4 h-4 text-cyan-400" />,
      action: () => {
        const p = SAMPLE_PRESETS.sample_multipage;
        if (p) onSelectSample?.(structuredClone(p));
      },
    });

    // View & Canvas
    if (onZoomFit) {
      list.push({
        id: 'zoom_fit',
        title: 'Fit Document to Window',
        category: 'View & Canvas',
        shortcut: 'F',
        icon: <Maximize2 className="w-4 h-4 text-cyan-400" />,
        action: onZoomFit,
      });
    }
    if (onZoomIn) {
      list.push({
        id: 'zoom_in',
        title: 'Zoom In (+20%)',
        category: 'View & Canvas',
        shortcut: '+',
        icon: <ZoomIn className="w-4 h-4 text-slate-300" />,
        action: onZoomIn,
      });
    }
    if (onZoomOut) {
      list.push({
        id: 'zoom_out',
        title: 'Zoom Out (-20%)',
        category: 'View & Canvas',
        shortcut: '-',
        icon: <ZoomOut className="w-4 h-4 text-slate-300" />,
        action: onZoomOut,
      });
    }
    if (onRotate) {
      list.push({
        id: 'rotate',
        title: 'Rotate 90° Clockwise',
        category: 'View & Canvas',
        shortcut: 'R',
        icon: <RotateCw className="w-4 h-4 text-slate-300" />,
        action: onRotate,
      });
    }
    if (onToggleHeatmap) {
      list.push({
        id: 'toggle_heatmap',
        title: 'Toggle Confidence Heatmap Overlay',
        category: 'View & Canvas',
        shortcut: 'H',
        icon: <Eye className="w-4 h-4 text-emerald-400" />,
        action: onToggleHeatmap,
      });
    }
    if (onToggleBBoxes) {
      list.push({
        id: 'toggle_bboxes',
        title: 'Toggle Bounding Box Vectors',
        category: 'View & Canvas',
        shortcut: 'B',
        icon: <Layers className="w-4 h-4 text-indigo-400" />,
        action: onToggleBBoxes,
      });
    }
    if (onToggleLoupe) {
      list.push({
        id: 'toggle_loupe',
        title: 'Toggle Retina Loupe Magnifier (3x)',
        category: 'View & Canvas',
        shortcut: 'Z',
        icon: <Sliders className="w-4 h-4 text-cyan-400" />,
        action: onToggleLoupe,
      });
    }
    if (onToggleInvert) {
      list.push({
        id: 'toggle_invert',
        title: 'Toggle Darkroom Blueprint Inversion',
        category: 'View & Canvas',
        shortcut: 'I',
        icon: <Sliders className="w-4 h-4 text-purple-400" />,
        action: onToggleInvert,
      });
    }

    // Analysis & Review
    if (onStartSpeedReview) {
      list.push({
        id: 'speed_review',
        title: 'Start Speed Review on Low-Confidence Words',
        category: 'Analysis & Review',
        shortcut: 'S',
        icon: <FastForward className="w-4 h-4 text-amber-400" />,
        action: onStartSpeedReview,
      });
    }

    // Export & Share
    if (onCopyText) {
      list.push({
        id: 'copy_text',
        title: 'Copy Transcribed Text to Clipboard',
        category: 'Export & Share',
        shortcut: '⌘C',
        icon: <Copy className="w-4 h-4 text-emerald-400" />,
        action: onCopyText,
      });
    }
    if (onExportTxt) {
      list.push({
        id: 'export_txt',
        title: 'Download Transcription Plain Text (.txt)',
        category: 'Export & Share',
        shortcut: '⌘E',
        icon: <Download className="w-4 h-4 text-indigo-400" />,
        action: onExportTxt,
      });
    }
    if (onExportJson) {
      list.push({
        id: 'export_json',
        title: 'Download Full OCR JSON Geometry (.json)',
        category: 'Export & Share',
        shortcut: '⌘J',
        icon: <Download className="w-4 h-4 text-purple-400" />,
        action: onExportJson,
      });
    }
    if (onExportCsv) {
      list.push({
        id: 'export_csv',
        title: 'Download Token Table (.csv)',
        category: 'Export & Share',
        shortcut: '⌘U',
        icon: <Download className="w-4 h-4 text-emerald-400" />,
        action: onExportCsv,
      });
    }

    return list;
  }, [
    onSelectSample,
    onZoomIn,
    onZoomOut,
    onZoomFit,
    onRotate,
    onToggleHeatmap,
    onToggleBBoxes,
    onToggleLoupe,
    onStartSpeedReview,
    onExportTxt,
    onExportJson,
    onExportCsv,
    onCopyText,
    onToggleInvert,
  ]);

  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    const q = query.toLowerCase();
    return commands.filter(
      (c) => c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q)
    );
  }, [commands, query]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [filtered]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, filtered.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev === 0 ? filtered.length - 1 : prev - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filtered[selectedIndex]) {
        filtered[selectedIndex].action();
        onClose();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      data-testid="command-palette-backdrop"
      onClick={onClose}
      className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-md flex items-start justify-center pt-20 sm:pt-28 px-4 animate-in fade-in duration-150"
    >
      <div
        data-testid="command-palette-modal"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl rounded-3xl bg-slate-900/90 border border-white/10 shadow-2xl shadow-black/80 backdrop-blur-2xl overflow-hidden flex flex-col animate-in zoom-in-95 duration-150"
      >
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-800/80">
          <Search className="w-5 h-5 text-cyan-400 flex-shrink-0" />
          <input
            ref={inputRef}
            data-testid="command-palette-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command, preset, or action..."
            className="flex-1 bg-transparent text-white placeholder-slate-400 text-sm sm:text-base outline-none font-medium"
          />
          <div className="flex items-center gap-1 text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-lg border border-slate-700/60">
            <span>Esc</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div data-testid="command-palette-list" className="max-h-80 overflow-y-auto p-2 space-y-1">
          {filtered.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-sm">
              No commands found for &ldquo;{query}&rdquo;
            </div>
          ) : (
            filtered.map((cmd, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={cmd.id}
                  type="button"
                  data-testid={`command-item-${cmd.id}`}
                  onClick={() => {
                    cmd.action();
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-left transition-all ${
                    isSelected
                      ? "bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-white border border-cyan-500/30 shadow-md"
                      : "text-slate-300 hover:bg-slate-800/60"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-1.5 rounded-xl bg-slate-800/80 border border-slate-700/60">
                      {cmd.icon}
                    </div>
                    <div>
                      <span className="text-sm font-semibold tracking-tight">{cmd.title}</span>
                      <span className="text-[10px] font-mono text-slate-500 block">{cmd.category}</span>
                    </div>
                  </div>
                  {cmd.shortcut && (
                    <kbd className="px-2 py-1 rounded-lg bg-slate-800/80 text-[11px] font-mono font-bold text-slate-300 border border-slate-700/60 shadow-sm">
                      {cmd.shortcut}
                    </kbd>
                  )}
                </button>
              );
            })
          )}
        </div>

        {/* Footer info strip */}
        <div className="px-4 py-2.5 border-t border-slate-800/80 bg-slate-950/60 flex items-center justify-between text-[11px] font-mono text-slate-500">
          <div className="flex items-center gap-2">
            <span>↑↓ Navigate</span>
            <span>•</span>
            <span>↵ Execute</span>
          </div>
          <div className="flex items-center gap-1.5 text-cyan-400 font-medium">
            <Sparkles className="w-3 h-3" />
            <span>Apple Silicon MPS • TrOCR-Large 558M</span>
          </div>
        </div>
      </div>
    </div>
  );
};
