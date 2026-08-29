'use client';

import React from 'react';
import { Sliders, Sun, Contrast, RotateCcw, Sparkles, Moon } from 'lucide-react';

export interface DarkroomSettings {
  contrast: number; // 0.5 to 3.0 (default 1.0)
  brightness: number; // 0.5 to 2.0 (default 1.0)
  grayscale: boolean;
  inverted: boolean;
}

export interface DarkroomToolbarProps {
  settings: DarkroomSettings;
  onChange: (newSettings: DarkroomSettings) => void;
  className?: string;
}

export const DarkroomToolbar: React.FC<DarkroomToolbarProps> = ({
  settings,
  onChange,
  className = '',
}) => {
  const handlePreset = (name: 'original' | 'faded' | 'pencil' | 'blueprint' | 'laser') => {
        switch (name) {
      case 'original':
        onChange({ contrast: 1.0, brightness: 1.0, grayscale: false, inverted: false });
        break;
      case 'faded':
        onChange({ contrast: 2.2, brightness: 1.15, grayscale: false, inverted: false });
        break;
      case 'pencil':
        onChange({ contrast: 1.8, brightness: 0.95, grayscale: true, inverted: false });
        break;
      case 'blueprint':
        onChange({ contrast: 1.4, brightness: 1.05, grayscale: false, inverted: true });
        break;
      case 'laser':
        onChange({ contrast: 2.8, brightness: 1.2, grayscale: true, inverted: false });
        break;
    }
  };

  return (
    <div
      data-testid="darkroom-toolbar-container"
      className={`p-3 rounded-2xl bg-slate-900/90 backdrop-blur-2xl border border-white/10 shadow-xl flex flex-wrap items-center gap-3 text-xs ${className}`}
    >
      <div className="flex items-center gap-1.5 text-slate-300 font-semibold pr-1">
        <Sliders className="w-4 h-4 text-cyan-400" />
        <span className="hidden sm:inline">Darkroom Studio</span>
      </div>

      {/* Preset Pills */}
      <div className="flex items-center gap-1">
        {[
          { id: 'original' as const, label: 'Natural' },
          { id: 'faded' as const, label: 'Faded Ink Boost' },
          { id: 'pencil' as const, label: 'Pencil B&W' },
          { id: 'blueprint' as const, label: 'Blueprint' },
        ].map((p) => (
          <button
            key={p.id}
            type="button"
            data-testid={`btn-preset-${p.id}`}
            onClick={() => handlePreset(p.id)}
            className="px-2.5 py-1 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors text-[11px] font-medium"
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-slate-800 hidden sm:block" />

      {/* Contrast Slider */}
      <div className="flex items-center gap-1.5">
        <Contrast className="w-3.5 h-3.5 text-slate-400" />
        <span className="text-[11px] text-slate-400 font-mono">{(settings.contrast).toFixed(1)}x</span>
        <input
          type="range"
          min="0.5"
          max="3.0"
          step="0.1"
          value={settings.contrast}
          onChange={(e) => {
            onChange({ ...settings, contrast: parseFloat(e.target.value) });
          }}
          className="w-16 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          title="Adjust Contrast"
        />
      </div>

      {/* Brightness Slider */}
      <div className="flex items-center gap-1.5">
        <Sun className="w-3.5 h-3.5 text-slate-400" />
        <span className="text-[11px] text-slate-400 font-mono">{(settings.brightness).toFixed(1)}x</span>
        <input
          type="range"
          min="0.5"
          max="2.0"
          step="0.05"
          value={settings.brightness}
          onChange={(e) => {
            onChange({ ...settings, brightness: parseFloat(e.target.value) });
          }}
          className="w-16 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
          title="Adjust Brightness"
        />
      </div>

      {/* Invert Button */}
      <button
        type="button"
        data-testid="btn-toggle-darkroom-invert"
        onClick={() => {
                    onChange({ ...settings, inverted: !settings.inverted });
        }}
        className={`p-1.5 rounded-xl transition-colors ${
          settings.inverted
            ? 'bg-purple-950/80 text-purple-300 border border-purple-800/60'
            : 'bg-slate-800 text-slate-400 hover:text-white'
        }`}
        title="Toggle Blueprint Invert"
      >
        <Moon className="w-3.5 h-3.5" />
      </button>

      {/* Reset Button */}
      <button
        type="button"
        data-testid="btn-darkroom-reset"
        onClick={() => handlePreset('original')}
        className="p-1.5 rounded-xl bg-slate-800 text-slate-400 hover:text-white transition-colors"
        title="Reset Darkroom"
      >
        <RotateCcw className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};
