'use client';

import React, { useState, useRef, useCallback } from 'react';
import { Split, Sparkles, FileText, MoveHorizontal } from 'lucide-react';
import { PageResult, LineItem } from '../types/ocr';
import { bboxToSvgRect } from '../lib/transformUtils';

export interface SplitCurtainProps {
  page: PageResult;
  className?: string;
}

export const SplitCurtain: React.FC<SplitCurtainProps> = ({ page, className = '' }) => {
  const [splitPercent, setSplitPercent] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const docWidth = page.width || 800;
  const docHeight = page.height || 1100;

  const updateSplitFromClientX = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const relativeX = clientX - rect.left;
    const clampedPercent = Math.max(5, Math.min(95, (relativeX / rect.width) * 100));
    setSplitPercent(Math.round(clampedPercent));
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    updateSplitFromClientX(e.clientX);
      };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    updateSplitFromClientX(e.clientX);
  };

  const handleMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
          }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches && e.touches[0]) {
      updateSplitFromClientX(e.touches[0].clientX);
    }
  };

  return (
    <div
      data-testid="split-curtain-container"
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchMove={handleTouchMove}
      className={`relative w-full h-full flex items-center justify-center bg-slate-950 overflow-hidden select-none cursor-ew-resize rounded-2xl border border-white/10 shadow-2xl ${className}`}
    >
      {/* Centered Document Sheet Box */}
      <div
        style={{
          width: `${docWidth}px`,
          height: `${docHeight}px`,
          maxWidth: '90%',
          maxHeight: '90%',
          aspectRatio: `${docWidth} / ${docHeight}`,
        }}
        className="relative bg-white rounded-xl shadow-2xl overflow-hidden pointer-events-none"
      >
        {/* Layer 1: Scanned Handwriting Image (Background) */}
        {page.image_url ? (
          <img
            src={page.image_url}
            alt="Original Handwritten Scan"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            draggable={false}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-100 text-slate-400 font-mono text-xs">
            [Original Document Image]
          </div>
        )}

        {/* Layer 2: Typeset OCR Overlay with Clip-Path Reveal */}
        <div
          data-testid="split-typeset-overlay"
          style={{
            clipPath: `polygon(${splitPercent}% 0, 100% 0, 100% 100%, ${splitPercent}% 100%)`,
          }}
          className="absolute inset-0 w-full h-full bg-slate-950/90 backdrop-blur-sm text-slate-100 flex flex-col p-6 overflow-hidden transition-none pointer-events-none"
        >
          {/* SVG Vector Overlays */}
          <svg
            viewBox={`0 0 ${docWidth} ${docHeight}`}
            className="absolute inset-0 w-full h-full pointer-events-none"
          >
            {page.lines.map((line: LineItem) => {
              const rect = bboxToSvgRect(line.bbox, docWidth, docHeight);
              return (
                <g key={line.line_id}>
                  <rect
                    x={rect.x}
                    y={rect.y}
                    width={rect.width}
                    height={rect.height}
                    rx={3}
                    fill="rgba(6, 182, 212, 0.12)"
                    stroke="#06b6d4"
                    strokeWidth={1.5}
                  />
                  <text
                    x={rect.x + 8}
                    y={rect.y + rect.height * 0.72}
                    fontSize={Math.max(12, rect.height * 0.55)}
                    fill="#38bdf8"
                    fontFamily="Inter, sans-serif"
                    fontWeight="600"
                  >
                    {line.text}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Vertical Divider Glass Line & Pill */}
        <div
          data-testid="split-curtain-divider"
          style={{ left: `${splitPercent}%` }}
          className="absolute top-0 bottom-0 w-1 bg-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.8)] -translate-x-1/2 z-30 pointer-events-auto"
        >
          {/* Centered Glass Grip Pill */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-12 rounded-2xl bg-slate-900/95 border-2 border-cyan-400 shadow-2xl flex items-center justify-center text-cyan-400">
            <MoveHorizontal className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Top Banner Mode Indicators */}
      <div className="absolute top-4 left-4 z-40 flex items-center gap-2 px-3 py-1.5 rounded-2xl bg-slate-900/90 backdrop-blur-xl border border-white/10 shadow-xl text-xs">
        <div className="flex items-center gap-1.5 text-slate-300 font-medium">
          <FileText className="w-3.5 h-3.5 text-slate-400" />
          <span>Original Scan ({splitPercent}%)</span>
        </div>
        <span className="text-slate-600">|</span>
        <div className="flex items-center gap-1.5 text-cyan-300 font-semibold">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Digital Typeset ({100 - splitPercent}%)</span>
        </div>
      </div>

      {/* Drag instruction pill */}
      <div className="absolute bottom-4 z-40 text-[11px] font-mono text-slate-400 px-3 py-1 rounded-full bg-slate-900/80 border border-white/10 backdrop-blur-md">
        Drag curtain divider to compare ink with neural OCR
      </div>
    </div>
  );
};
