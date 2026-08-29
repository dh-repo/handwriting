'use client';

import React from 'react';
import { Keyboard, X, Sparkles, Command } from 'lucide-react';

export interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutSection {
  title: string;
  items: { key: string; label: string }[];
}

const SHORTCUT_SECTIONS: ShortcutSection[] = [
  {
    title: 'Global & Navigation',
    items: [
      { key: '⌘K / /', label: 'Open Command Palette' },
      { key: '?', label: 'Open Keyboard Shortcuts' },
      { key: 'Esc', label: 'Close Modal / Deselect' },
      { key: '⌘1 - ⌘5', label: 'Quick-Load Demo Presets' },
    ],
  },
  {
    title: 'Canvas & Viewport',
    items: [
      { key: 'F', label: 'Fit Document to Screen' },
      { key: 'W', label: 'Fit to Width' },
      { key: '+ / -', label: 'Zoom In / Zoom Out' },
      { key: 'R', label: 'Rotate 90° Clockwise' },
      { key: '0', label: 'Reset View (100%)' },
      { key: 'H', label: 'Toggle Heatmap' },
      { key: 'B', label: 'Toggle Bounding Boxes' },
      { key: 'Z', label: 'Toggle Retina Loupe (3x)' },
      { key: 'I', label: 'Darkroom Blueprint Invert' },
    ],
  },
  {
    title: 'Inline Editor & Speed Review',
    items: [
      { key: 'S', label: 'Switch to Speed Review' },
      { key: '1 - 5', label: 'Quick-Pick Suggestion' },
      { key: 'Enter', label: 'Accept & Next Word' },
      { key: 'Tab / ⇧Tab', label: 'Skip / Previous Word' },
      { key: '⌘Z / ⌘Y', label: 'Undo / Redo Line Edits' },
      { key: 'V', label: 'VoiceOver Read Aloud' },
    ],
  },
  {
    title: 'Export & Share',
    items: [
      { key: '⌘C', label: 'Copy Clean Text' },
      { key: '⌘E', label: 'Download Plain Text (.txt)' },
      { key: '⌘J', label: 'Download OCR JSON' },
      { key: '⌘U', label: 'Download Token Table (.csv)' },
    ],
  },
];

export const ShortcutsModal: React.FC<ShortcutsModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div
      data-testid="shortcuts-modal-backdrop"
      onClick={onClose}
      className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-150"
    >
      <div
        data-testid="shortcuts-modal-content"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-2xl rounded-3xl bg-slate-900/90 border border-white/10 shadow-2xl shadow-black/80 backdrop-blur-2xl p-6 space-y-6 animate-in zoom-in-95 duration-150"
      >
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              <Keyboard className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <span>Keyboard Shortcuts</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                  Mac Pro
                </span>
              </h3>
              <p className="text-xs text-slate-400">Pro hotkeys for ultra-fast document transcription and review</p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[60vh] overflow-y-auto pr-1">
          {SHORTCUT_SECTIONS.map((sec) => (
            <div
              key={sec.title}
              className="p-4 rounded-2xl bg-slate-950/60 border border-slate-800/80 space-y-2.5 shadow-inner"
            >
              <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-wider font-mono">
                {sec.title}
              </h4>
              <div className="space-y-1.5">
                {sec.items.map((item) => (
                  <div key={item.key} className="flex items-center justify-between text-xs">
                    <span className="text-slate-300">{item.label}</span>
                    <kbd className="px-2 py-0.5 rounded-lg bg-slate-800 text-[11px] font-mono font-bold text-slate-200 border border-slate-700/60 shadow-sm">
                      {item.key}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-xs text-slate-400 font-mono">
          <span>Tip: Press ⌘K anywhere to search commands</span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
