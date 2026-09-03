'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCw,
  RefreshCw,
  Eye,
  EyeOff,
  Layers,
  FileText,
  MoveHorizontal,
  Search,
  Sliders,
  Sparkles,
  Crosshair,
} from 'lucide-react';
import { PageResult, LineItem, WordToken, WordCandidate } from '../types/ocr';
import { bboxToSvgRect } from '../lib/transformUtils';
import { getConfidenceColor } from '../lib/colorUtils';

export interface DocumentViewerProps {
  page: PageResult;
  allPages?: PageResult[];
  activePageIndex?: number;
  onPageChange?: (pageIndex: number) => void;
  selectedLineId?: string | null;
  selectedWordId?: string | null;
  hoveredLineId?: string | null;
  hoveredWordId?: string | null;
  onSelectLine?: (lineId: string) => void;
  onSelectWord?: (wordId: string, parentLineId?: string) => void;
  onHoverLine?: (lineId: string | null) => void;
  onHoverWord?: (wordId: string | null) => void;
  isLoupeActive?: boolean;
  onToggleLoupe?: () => void;
  isInverted?: boolean;
  onToggleInvert?: () => void;
  contrastBoost?: number;
  signatureLineIds?: string[];
  className?: string;
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  page,
  allPages = [page],
  activePageIndex = 0,
  onPageChange,
  selectedLineId,
  selectedWordId,
  hoveredLineId,
  hoveredWordId,
  onSelectLine,
  onSelectWord,
  onHoverLine,
  onHoverWord,
  isLoupeActive = false,
  onToggleLoupe,
  isInverted = false,
  onToggleInvert,
  contrastBoost = 1.0,
  signatureLineIds = [],
  className = '',
}) => {
  const [scale, setScale] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [rotation, setRotation] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Touch gesture pinch state
  const touchDistanceRef = useRef<number | null>(null);
  const touchStartScaleRef = useRef<number>(1.0);

  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showConfidenceHeatmap, setShowConfidenceHeatmap] = useState<boolean>(true);
  const [showWordBoxes, setShowWordBoxes] = useState<boolean>(true);
  const [showSidebar, setShowSidebar] = useState<boolean>(allPages.length > 1);

  // Loupe pointer state
  const [loupePos, setLoupePos] = useState<{ x: number; y: number; docX: number; docY: number } | null>(null);

  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    text: string;
    confidence: number;
    lineIndex: number;
    wordIndex?: number;
    alternatives?: (WordCandidate | string)[];
  } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const docWidth = page.width || 800;
  const docHeight = page.height || 1100;

  const handleZoom = useCallback((delta: number) => {
    setScale((prev) => Math.min(5.0, Math.max(0.2, +(prev + delta).toFixed(2))));
  }, []);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      setScale(+Math.min(5.0, Math.max(0.2, val)).toFixed(2));
    }
  };

  const handleZoomFit = useCallback(() => {
    if (!containerRef.current || !docWidth || !docHeight) {
      setScale(1.0);
      setPan({ x: 0, y: 0 });
      return;
    }
    const { clientWidth, clientHeight } = containerRef.current;
    const padding = 32;
    const scaleX = (clientWidth - padding) / docWidth;
    const scaleY = (clientHeight - padding) / docHeight;
    const fitScale = Math.min(scaleX, scaleY, 1.2);
    setScale(+fitScale.toFixed(2));
    setPan({ x: 0, y: 0 });
  }, [docWidth, docHeight]);

  const handleZoomFitWidth = useCallback(() => {
    if (!containerRef.current || !docWidth) {
      setScale(1.0);
      setPan({ x: 0, y: 0 });
      return;
    }
    const { clientWidth } = containerRef.current;
    const padding = 32;
    const fitScale = Math.min((clientWidth - padding) / docWidth, 2.5);
    setScale(+fitScale.toFixed(2));
    setPan({ x: 0, y: 0 });
  }, [docWidth]);

  const handleReset = useCallback(() => {
    setScale(1.0);
    setPan({ x: 0, y: 0 });
    setRotation(0);
  }, []);

  const handleFocusCrop = useCallback(() => {
    if (!selectedWordId || !containerRef.current || !docWidth || !docHeight) return;
    for (const line of page.lines) {
      const word = line.words?.find((w) => w.word_id === selectedWordId);
      if (word && word.bbox) {
        const [ymin, xmin, ymax, xmax] = word.bbox;
        const wordCenterX = ((xmin + xmax) / 2) * docWidth;
        const wordCenterY = ((ymin + ymax) / 2) * docHeight;

        const targetScale = Math.max(scale, 1.8);
        const docCenterX = docWidth / 2;
        const docCenterY = docHeight / 2;
        const targetPanX = (docCenterX - wordCenterX) * targetScale;
        const targetPanY = (docCenterY - wordCenterY) * targetScale;

        setScale(+targetScale.toFixed(2));
        setPan({ x: Math.round(targetPanX), y: Math.round(targetPanY) });
        break;
      }
    }
  }, [selectedWordId, docWidth, docHeight, page.lines, scale]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      setScale((prev) => Math.min(5.0, Math.max(0.2, +(prev * zoomFactor).toFixed(2))));
    } else {
      setPan((prev) => ({
        x: prev.x - e.deltaX,
        y: prev.y - e.deltaY,
      }));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }

    if (isLoupeActive && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const clientX = e.clientX;
      const clientY = e.clientY;
      if (
        clientX >= rect.left &&
        clientX <= rect.right &&
        clientY >= rect.top &&
        clientY <= rect.bottom
      ) {
        const normX = (clientX - rect.left) / rect.width;
        const normY = (clientY - rect.top) / rect.height;
        setLoupePos({
          x: clientX,
          y: clientY,
          docX: normX * docWidth,
          docY: normY * docHeight,
        });
      } else {
        setLoupePos(null);
      }
    } else {
      if (loupePos !== null) setLoupePos(null);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches && e.touches.length === 2) {
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      touchDistanceRef.current = dist;
      touchStartScaleRef.current = scale;
    } else if (e.touches && e.touches.length === 1) {
      setIsDragging(true);
      setDragStart({ x: e.touches[0].clientX - pan.x, y: e.touches[0].clientY - pan.y });
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches && e.touches.length === 2 && touchDistanceRef.current !== null) {
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      if (touchDistanceRef.current > 0) {
        const ratio = dist / touchDistanceRef.current;
        const newScale = Math.min(5.0, Math.max(0.2, +(touchStartScaleRef.current * ratio).toFixed(2)));
        setScale(newScale);
      }
    } else if (e.touches && e.touches.length === 1 && isDragging) {
      setPan({
        x: e.touches[0].clientX - dragStart.x,
        y: e.touches[0].clientY - dragStart.y,
      });
    }
  };

  const handleTouchEnd = () => {
    touchDistanceRef.current = null;
    setIsDragging(false);
  };

  return (
    <div
      data-testid="document-viewer-container"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      className={`relative flex h-full w-full bg-slate-950 text-slate-100 select-none overflow-hidden rounded-2xl border border-slate-800/80 shadow-xl ${className}`}
    >
      {/* Multi-Page Sidebar Thumbnail Strip */}
      {showSidebar && allPages.length > 1 && (
        <aside
          data-testid="multipage-sidebar"
          className="w-24 bg-slate-900/90 backdrop-blur border-r border-slate-800 flex flex-col items-center py-4 px-2 space-y-3 z-20 overflow-y-auto"
        >
          <div className="flex items-center justify-between w-full px-1">
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Pages</span>
            <span className="text-[10px] font-mono text-slate-500">{allPages.length}</span>
          </div>
          {allPages.map((p, idx) => (
            <button
              key={idx}
              type="button"
              data-testid={`page-thumbnail-${idx + 1}`}
              onClick={() => onPageChange?.(idx)}
              className={`w-full rounded-xl p-1.5 border transition-all text-center flex flex-col items-center gap-1.5 ${
                activePageIndex === idx
                  ? "border-cyan-500 bg-cyan-950/60 shadow-md ring-2 ring-cyan-500/40"
                  : "border-slate-800 hover:border-slate-700 bg-slate-950/40 opacity-70 hover:opacity-100"
              }`}
            >
              <div className="w-16 h-20 bg-slate-800 rounded-lg flex items-center justify-center overflow-hidden border border-slate-700 relative">
                {p.image_url ? (
                  <img src={p.image_url} alt={`Page ${idx + 1}`} className="w-full h-full object-cover" />
                ) : (
                  <FileText className="w-6 h-6 text-slate-500" />
                )}
              </div>
              <span className="text-[11px] font-medium text-slate-300">Page {idx + 1}</span>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold ${
                  p.mean_confidence >= 0.90
                    ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                    : p.mean_confidence >= 0.70
                    ? "bg-amber-950 text-amber-300 border border-amber-800"
                    : "bg-rose-950 text-rose-300 border border-rose-800"
                }`}
              >
                {(p.mean_confidence * 100).toFixed(0)}%
              </span>
            </button>
          ))}
        </aside>
      )}

      {/* Main Viewport Container */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`relative flex-1 h-full w-full flex items-center justify-center overflow-hidden cursor-${
          isDragging ? "grabbing" : "grab"
        }`}
      >
        {/* Floating Viewport Toolbar */}
        <div
          data-testid="viewer-toolbar"
          className="absolute top-4 right-4 z-30 flex items-center gap-1 p-1.5 rounded-2xl bg-slate-900/90 backdrop-blur-2xl border border-white/10 shadow-2xl shadow-black/80"
        >
          <button
            type="button"
            data-testid="btn-zoom-in"
            onClick={() => handleZoom(0.2)}
            title="Zoom In (+20%)"
            className="p-2 rounded-xl text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <span
            data-testid="zoom-level-text"
            className="text-xs font-mono font-medium text-slate-300 px-1 min-w-[3.2rem] text-center"
          >
            {Math.round(scale * 100)}%
          </span>

          <button
            type="button"
            data-testid="btn-zoom-out"
            onClick={() => handleZoom(-0.2)}
            title="Zoom Out (-20%)"
            className="p-2 rounded-xl text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          {/* Continuous Zoom Slider */}
          <div className="hidden sm:flex items-center px-1">
            <input
              type="range"
              min="0.2"
              max="5.0"
              step="0.05"
              value={scale}
              onChange={handleSliderChange}
              data-testid="zoom-slider"
              title="Continuous Zoom Slider (0.2x - 5.0x)"
              className="w-20 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
          </div>

          <div className="w-px h-4 bg-slate-800 mx-0.5" />

          <button
            type="button"
            data-testid="btn-zoom-fit"
            onClick={handleZoomFit}
            title="Fit to Screen"
            className="p-2 rounded-xl text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors"
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          <button
            type="button"
            data-testid="btn-zoom-fit-width"
            onClick={handleZoomFitWidth}
            title="Fit to Width"
            className="p-2 rounded-xl text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors"
          >
            <MoveHorizontal className="w-4 h-4" />
          </button>

          <button
            type="button"
            data-testid="btn-reset-view"
            onClick={handleReset}
            title="Reset View (100%)"
            className="p-2 rounded-xl text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          {selectedWordId && (
            <button
              type="button"
              data-testid="btn-focus-crop"
              onClick={handleFocusCrop}
              title="Focus and zoom into selected handwriting crop"
              className="px-2.5 py-1.5 rounded-xl text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/30 transition-all shadow-sm shadow-cyan-500/20 flex items-center gap-1 animate-in fade-in"
            >
              <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
              <span className="hidden sm:inline">Focus Crop</span>
            </button>
          )}

          <button
            type="button"
            data-testid="btn-rotate"
            onClick={() => setRotation((prev) => (prev + 90) % 360)}
            title="Rotate 90°"
            className="p-2 rounded-xl text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors"
          >
            <RotateCw className="w-4 h-4" />
          </button>

          <div className="w-px h-4 bg-slate-800 mx-0.5" />

          {/* Retina Loupe Toggle */}
          {onToggleLoupe && (
            <button
              type="button"
              data-testid="btn-toggle-loupe"
              onClick={onToggleLoupe}
              title={isLoupeActive ? "Disable Retina Loupe (Z)" : "Enable Retina Loupe (Z)"}
              className={`p-2 rounded-xl transition-colors ${
                isLoupeActive
                  ? "text-cyan-400 bg-cyan-950/80 border border-cyan-800/60 shadow-sm shadow-cyan-500/20"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/80"
              }`}
            >
              <Search className="w-4 h-4" />
            </button>
          )}

          {/* Blueprint Darkroom Invert Toggle */}
          {onToggleInvert && (
            <button
              type="button"
              data-testid="btn-toggle-invert"
              onClick={onToggleInvert}
              title={isInverted ? "Default Contrast" : "Darkroom Blueprint Invert"}
              className={`p-2 rounded-xl transition-colors ${
                isInverted
                  ? "text-purple-400 bg-purple-950/80 border border-purple-800/60 shadow-sm shadow-purple-500/20"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/80"
              }`}
            >
              <Sliders className="w-4 h-4" />
            </button>
          )}

          <button
            type="button"
            data-testid="btn-toggle-bboxes"
            onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
            title={showBoundingBoxes ? "Hide Bounding Boxes" : "Show Bounding Boxes"}
            className={`p-2 rounded-xl transition-colors ${
              showBoundingBoxes
                ? "text-indigo-400 bg-indigo-950/80 border border-indigo-800/60"
                : "text-slate-400 hover:text-white hover:bg-slate-800/80"
            }`}
          >
            <Layers className="w-4 h-4" />
          </button>

          <button
            type="button"
            data-testid="btn-toggle-heatmap"
            onClick={() => setShowConfidenceHeatmap(!showConfidenceHeatmap)}
            title={showConfidenceHeatmap ? "Disable Heatmap" : "Enable Heatmap"}
            className={`p-2 rounded-xl transition-colors ${
              showConfidenceHeatmap
                ? "text-emerald-400 bg-emerald-950/80 border border-emerald-800/60"
                : "text-slate-400 hover:text-white hover:bg-slate-800/80"
            }`}
          >
            {showConfidenceHeatmap ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>

        {/* Transformed Document Canvas & SVG Vector Layer */}
        <div
          ref={canvasRef}
          data-testid="viewer-canvas-wrapper"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale}) rotate(${rotation}deg)`,
            transformOrigin: "center center",
            transition: isDragging ? "none" : "transform 0.1s ease-out",
            width: `${docWidth}px`,
            height: `${docHeight}px`,
            filter: isInverted
              ? "invert(1) hue-rotate(180deg) contrast(1.35) brightness(1.05)"
              : contrastBoost > 1
              ? `contrast(${contrastBoost}) brightness(1.02)`
              : "none",
          }}
          className="relative shadow-2xl rounded-2xl bg-white overflow-hidden"
        >
          {/* Base Document Image */}
          {page.image_url ? (
            <img
              src={page.image_url}
              alt={`Handwriting Document Page ${page.page_number}`}
              className="w-full h-full object-contain pointer-events-none select-none"
              draggable={false}
            />
          ) : (
            <div className="w-full h-full bg-slate-100 dark:bg-slate-900 flex items-center justify-center text-slate-400 font-mono">
              [No Image Preview Available]
            </div>
          )}

          {/* SVG Vector Bounding Box Overlay */}
          {showBoundingBoxes && (
            <svg
              data-testid="svg-bbox-layer"
              viewBox={`0 0 ${docWidth} ${docHeight}`}
              className="absolute inset-0 w-full h-full pointer-events-none"
            >
              {page.lines.map((line: LineItem, lineIdx: number) => {
                const isLineSelected = selectedLineId === line.line_id;
                const isLineHovered = hoveredLineId === line.line_id;
                const isSignatureCandidate = signatureLineIds.includes(line.line_id);
                const colorStyle = getConfidenceColor(line.confidence);
                const rect = bboxToSvgRect(line.bbox, docWidth, docHeight);
                const stroke = isLineSelected
                  ? '#06b6d4'
                  : isLineHovered
                    ? '#38bdf8'
                    : isSignatureCandidate
                      ? '#f59e0b'
                      : colorStyle.stroke;

                return (
                  <g key={line.line_id} data-testid={`svg-line-group-${line.line_id}`} className="pointer-events-auto">
                    {/* Line Bounding Box */}
                    <rect
                      data-testid={`svg-line-rect-${line.line_id}`}
                      data-signature-candidate={isSignatureCandidate ? 'true' : 'false'}
                      x={rect.x}
                      y={rect.y}
                      width={rect.width}
                      height={rect.height}
                      rx={4}
                      fill={showConfidenceHeatmap ? colorStyle.fill : "rgba(99, 102, 241, 0.08)"}
                      stroke={stroke}
                      strokeWidth={isLineSelected ? 3 : isLineHovered || isSignatureCandidate ? 2.5 : 1.5}
                      strokeDasharray={line.confidence < 0.70 ? "4 2" : undefined}
                      className="cursor-pointer transition-all duration-150"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectLine?.(line.line_id);
                      }}
                      onMouseEnter={() => {
                        onHoverLine?.(line.line_id);
                        setTooltip({
                          visible: true,
                          x: rect.x + rect.width / 2,
                          y: Math.max(20, rect.y - 8),
                          text: line.text,
                          confidence: line.confidence,
                          lineIndex: lineIdx + 1,
                        });
                      }}
                      onMouseLeave={() => {
                        onHoverLine?.(null);
                        setTooltip(null);
                      }}
                    />

                    {/* Word-level Bounding Boxes */}
                    {showWordBoxes &&
                      line.words?.map((word: WordToken, wIdx: number) => {
                        const isWordSelected = selectedWordId === word.word_id;
                        const isWordHovered = hoveredWordId === word.word_id;
                        const isLowConfidence = word.confidence < 0.85;
                        const wordRect = bboxToSvgRect(word.bbox, docWidth, docHeight);

                        return (
                          <g key={word.word_id}>
                            <rect
                              data-testid={`svg-word-rect-${word.word_id}`}
                              data-confidence-low={isLowConfidence ? 'true' : 'false'}
                              x={wordRect.x}
                              y={wordRect.y}
                              width={wordRect.width}
                              height={wordRect.height}
                              rx={3}
                              fill={
                                isWordSelected
                                  ? "rgba(6, 182, 212, 0.35)"
                                  : isWordHovered
                                  ? "rgba(245, 158, 11, 0.3)"
                                  : isLowConfidence
                                  ? "rgba(245, 158, 11, 0.14)"
                                  : "transparent"
                              }
                              stroke={
                                isWordSelected
                                  ? "#06b6d4"
                                  : isWordHovered
                                  ? "#f59e0b"
                                  : isLowConfidence
                                  ? "#f59e0b"
                                  : "transparent"
                              }
                              strokeWidth={isWordSelected || isWordHovered ? 2 : isLowConfidence ? 1.5 : 1}
                              strokeDasharray={isLowConfidence && !isWordSelected && !isWordHovered ? "3 2" : undefined}
                              className="cursor-pointer transition-all duration-150"
                              onClick={(e) => {
                                e.stopPropagation();
                                onSelectWord?.(word.word_id, line.line_id);
                              }}
                              onMouseEnter={(e) => {
                                e.stopPropagation();
                                onHoverWord?.(word.word_id);
                                setTooltip({
                                  visible: true,
                                  x: wordRect.x + wordRect.width / 2,
                                  y: Math.max(20, wordRect.y - 6),
                                  text: word.text,
                                  confidence: word.confidence,
                                  lineIndex: lineIdx + 1,
                                  wordIndex: wIdx + 1,
                                  alternatives: word.alternatives || word.candidate_tokens,
                                });
                              }}
                              onMouseLeave={() => {
                                onHoverWord?.(null);
                              }}
                            />
                            {isWordSelected && (
                              <>
                                <rect
                                  data-testid={`svg-crop-halo-${word.word_id}`}
                                  x={wordRect.x - 3}
                                  y={wordRect.y - 3}
                                  width={wordRect.width + 6}
                                  height={wordRect.height + 6}
                                  rx={5}
                                  fill="none"
                                  stroke="#38bdf8"
                                  strokeWidth={1.5}
                                  className="pointer-events-none"
                                />
                                <path
                                  d={`M ${wordRect.x - 5} ${wordRect.y + 3} L ${wordRect.x - 5} ${wordRect.y - 5} L ${wordRect.x + 3} ${wordRect.y - 5}`}
                                  stroke="#06b6d4"
                                  strokeWidth={2}
                                  fill="none"
                                  className="pointer-events-none"
                                />
                                <path
                                  d={`M ${wordRect.x + wordRect.width + 5} ${wordRect.y + 3} L ${wordRect.x + wordRect.width + 5} ${wordRect.y - 5} L ${wordRect.x + wordRect.width - 3} ${wordRect.y - 5}`}
                                  stroke="#06b6d4"
                                  strokeWidth={2}
                                  fill="none"
                                  className="pointer-events-none"
                                />
                                <path
                                  d={`M ${wordRect.x - 5} ${wordRect.y + wordRect.height - 3} L ${wordRect.x - 5} ${wordRect.y + wordRect.height + 5} L ${wordRect.x + 3} ${wordRect.y + wordRect.height + 5}`}
                                  stroke="#06b6d4"
                                  strokeWidth={2}
                                  fill="none"
                                  className="pointer-events-none"
                                />
                                <path
                                  d={`M ${wordRect.x + wordRect.width + 5} ${wordRect.y + wordRect.height - 3} L ${wordRect.x + wordRect.width + 5} ${wordRect.y + wordRect.height + 5} L ${wordRect.x + wordRect.width - 3} ${wordRect.y + wordRect.height + 5}`}
                                  stroke="#06b6d4"
                                  strokeWidth={2}
                                  fill="none"
                                  className="pointer-events-none"
                                />
                              </>
                            )}
                          </g>
                        );
                      })}
                  </g>
                );
              })}
            </svg>
          )}

          {/* Tethered Hover Tooltip */}
          {tooltip && tooltip.visible && !isLoupeActive && (
            <div
              data-testid="viewer-tooltip"
              style={{
                left: `${tooltip.x}px`,
                top: `${tooltip.y}px`,
                transform: "translate(-50%, -100%)",
              }}
              className="absolute pointer-events-none z-50 px-3.5 py-2 rounded-2xl bg-slate-950/95 backdrop-blur-xl border border-white/15 shadow-2xl text-xs text-white flex flex-col gap-1 animate-in fade-in"
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-slate-400">
                  L{tooltip.lineIndex}
                  {tooltip.wordIndex !== undefined ? ` W${tooltip.wordIndex}` : ""}
                </span>
                <span className="font-bold max-w-[200px] truncate">{tooltip.text}</span>
                <span
                  className={`font-mono font-bold px-1.5 py-0.5 rounded-md text-[10px] ${
                    tooltip.confidence >= 0.90
                      ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                      : tooltip.confidence >= 0.70
                      ? "bg-amber-950 text-amber-300 border border-amber-800"
                      : "bg-rose-950 text-rose-300 border border-rose-800"
                  }`}
                >
                  {(tooltip.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {/* Render candidate alternatives if available */}
              {tooltip.alternatives && tooltip.alternatives.length > 0 && (
                <div data-testid="tooltip-alternatives" className="text-[10px] text-slate-400 border-t border-slate-800 pt-1">
                  <span className="font-semibold text-cyan-400">Alternatives: </span>
                  {tooltip.alternatives.slice(0, 3).map((alt, aIdx) => {
                    const altText = typeof alt === "string" ? alt : alt.text;
                    const altConf = typeof alt === "object" && alt.confidence !== undefined ? ` (${(alt.confidence * 100).toFixed(0)}%)` : "";
                    return (
                      <span key={aIdx} className="font-mono text-slate-300">
                        {altText}
                        {altConf}
                        {aIdx < Math.min(tooltip.alternatives!.length, 3) - 1 ? ", " : ""}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 3x Optical Retina Loupe Overlay */}
        {isLoupeActive && loupePos && page.image_url && (
          <div
            data-testid="retina-loupe-lens"
            style={{
              left: `${loupePos.x - 70}px`,
              top: `${loupePos.y - 70}px`,
              width: "140px",
              height: "140px",
            }}
            className="loupe-lens"
          >
            <div
              style={{
                width: `${docWidth * 3}px`,
                height: `${docHeight * 3}px`,
                transform: `translate(${-loupePos.docX * 3 + 70}px, ${-loupePos.docY * 3 + 70}px)`,
              }}
              className="relative bg-white"
            >
              <img
                src={page.image_url}
                alt="Magnified Stroke"
                className="w-full h-full object-contain"
              />
            </div>
            <div className="loupe-crosshair" />
            <div className="absolute bottom-1 right-2 text-[9px] font-mono font-bold text-cyan-400 bg-slate-950/80 px-1 rounded">
              3.0x
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
