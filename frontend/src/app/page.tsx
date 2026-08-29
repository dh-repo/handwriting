'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Image from 'next/image';
import {
  PenTool,
  Cpu,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  Zap,
  Activity,
  ChevronRight,
  BookOpen,
  Search,
  Keyboard,
  Sliders,
  Eye,
  Layers,
  FileText,
  Stethoscope,
  Split,
  Maximize,
  Scale,
  Sun,
  Contrast,
  SlidersHorizontal,
} from 'lucide-react';
import { useDocumentContext } from '../context/DocumentContext';
import { Dropzone } from '../components/Dropzone';
import { SampleGallery } from '../components/SampleGallery';
import { DocumentViewer } from '../components/DocumentViewer';
import { InlineEditor } from '../components/InlineEditor';
import { ExportToolbar } from '../components/ExportToolbar';
import { MetricsSummary } from '../components/MetricsSummary';
import { CommandPalette } from '../components/CommandPalette';
import { MedicalEntitiesCard } from '../components/MedicalEntitiesCard';
import { ShortcutsModal } from '../components/ShortcutsModal';
import { SplitCurtain } from '../components/SplitCurtain';
import { SignatureInspector } from '../components/SignatureInspector';
import { DarkroomToolbar, DarkroomSettings } from '../components/DarkroomToolbar';
import { DiagnosticsHUD } from '../components/DiagnosticsHUD';
import { SAMPLE_PRESETS, SAMPLE_LEGAL_CONTRACT } from '../lib/sampleDocuments';
import { DocumentOCRResult } from '../types/ocr';
import { apiClient } from '../lib/apiClient';
import {
  exportDocumentAsTxt,
  exportDocumentAsJson,
  exportDocumentAsCsv,
  copyTextToClipboard,
  downloadFile,
} from '../lib/exportUtils';

export type ViewMode = 'studio' | 'split_curtain' | 'split_diff' | 'clinical_radar' | 'signatures';

export default function WorkspacePage() {
  const {
    document,
    activePageIndex,
    activePage,
    setDocument,
    setActivePageIndex,
    selectedLineId,
    selectedWordId,
    hoveredLineId,
    hoveredWordId,
    setSelectedLineId,
    setSelectedWordId,
    setHoveredLineId,
    setHoveredWordId,
    isProcessing,
    uploadProgress,
    processingStage,
    error,
    setIsProcessing,
    setUploadProgress,
    setProcessingStage,
    setError,
    updateLineText,
    updateWordText,
    revertAll,
    undo,
    redo,
    canUndo,
    canRedo,
  } = useDocumentContext();

  const activeObjectUrlRef = useRef<string | null>(null);

  // Apple Pro Workspace States
  const [viewMode, setViewMode] = useState<ViewMode>('studio');
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState(false);
  const [isDarkroomOpen, setIsDarkroomOpen] = useState(false);

  // Darkroom settings
  const [darkroom, setDarkroom] = useState<DarkroomSettings>({
    contrast: 1.0,
    brightness: 1.0,
    grayscale: false,
    inverted: false,
  });

  const [isLoupeActive, setIsLoupeActive] = useState(false);

  const cleanupObjectURL = useCallback(() => {
    if (activeObjectUrlRef.current) {
      try {
        if (typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
          URL.revokeObjectURL(activeObjectUrlRef.current);
        }
      } catch {
        // ignore
      }
      activeObjectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      cleanupObjectURL();
    };
  }, [cleanupObjectURL]);

  // Global Keyboard Shortcuts (⌘K, ?, ⌘D, Z, I, etc.)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        setIsDiagnosticsOpen((prev) => !prev);
      } else if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        e.preventDefault();
        setIsCommandPaletteOpen(true);
      } else if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        e.preventDefault();
        setIsShortcutsOpen((prev) => !prev);
      } else if (e.key.toLowerCase() === 'z' && !e.metaKey && !e.ctrlKey && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        setIsLoupeActive((prev) => !prev);
      } else if (e.key.toLowerCase() === 'i' && !e.metaKey && !e.ctrlKey && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        setDarkroom((prev) => ({ ...prev, inverted: !prev.inverted }));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleFileAccepted = async (file: File) => {
    setIsProcessing(true);
    setUploadProgress(20);
    setProcessingStage('Uploading document to recognition pipeline...');
    setError(null);

    try {
      setUploadProgress(50);
      setProcessingStage('Running TrOCR Vision-Encoder-Decoder inference & Lexicon rescoring...');

      const result: DocumentOCRResult = await apiClient.recognizeFile(file, {
        model_type: 'trocr-handwritten-mps-v1',
        rescore: true,
      });

      setUploadProgress(90);
      setProcessingStage('Finalizing transcription results...');

      if (file.type.startsWith('image/') && result.pages[0]) {
        try {
          cleanupObjectURL();
          const previewUrl = URL.createObjectURL(file);
          activeObjectUrlRef.current = previewUrl;
          result.pages[0].image_url = previewUrl;
        } catch {
          // fallback
        }
      }

      setUploadProgress(100);
      setProcessingStage('Done');
      setDocument(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to process document';
      setError(msg);
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
    }
  };

  const handleSelectSample = (sample: DocumentOCRResult) => {
    cleanupObjectURL();
    setDocument(sample);
  };

  const handleReset = () => {
    cleanupObjectURL();
    setDocument(null);
  };

  const handleSwitchMode = (mode: ViewMode) => {
    setViewMode(mode);
  };

  // Export helpers for Command Palette
  const handleExportTxt = () => {
    if (!document) return;
    const content = exportDocumentAsTxt(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_transcription.txt`;
    downloadFile(content, filename, 'text/plain;charset=utf-8');
  };

  const handleExportJson = () => {
    if (!document) return;
    const content = exportDocumentAsJson(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_ocr_result.json`;
    downloadFile(content, filename, 'application/json');
  };

  const handleExportCsv = () => {
    if (!document) return;
    const content = exportDocumentAsCsv(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_tokens.csv`;
    downloadFile(content, filename, 'text/csv;charset=utf-8');
  };

  const handleCopyText = async () => {
    if (!document) return;
    const content = exportDocumentAsTxt(document);
    await copyTextToClipboard(content);
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#08090d] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Apple Pro Studio Header Island */}
      <header className="h-16 border-b border-white/[0.08] bg-[#0d111c]/80 backdrop-blur-2xl px-4 sm:px-6 flex items-center justify-between z-30 flex-shrink-0 shadow-2xl shadow-black/50">
        <div className="flex items-center gap-3.5">
          <div className="relative w-10 h-10 rounded-2xl overflow-hidden shadow-lg shadow-cyan-500/20 border border-cyan-500/30 group">
            <Image
              src="/logo.png"
              alt="Handwriting AI Logo"
              width={40}
              height={40}
              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
              priority
            />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <span className="text-gradient font-extrabold">
                Handwriting Recognition AI
              </span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shadow-sm shadow-cyan-500/20">
                PRO SOTA
              </span>
            </h1>
            <p className="text-[11px] text-slate-400 hidden sm:flex items-center gap-2 font-medium">
              <span>TrOCR-Large (558M)</span>
              <span className="text-slate-600">•</span>
              <span className="text-cyan-400/90">Universal RxNorm Rescorer</span>
              <span className="text-slate-600">•</span>
              <span className="text-emerald-400 font-mono">1.04% CER</span>
            </p>
          </div>
        </div>

        {/* Center View Mode Switcher (When document is loaded) */}
        {document && (
          <div className="hidden lg:flex items-center p-1 rounded-2xl bg-slate-900/90 border border-white/10 shadow-inner gap-0.5">
            <button
              type="button"
              data-testid="btn-tab-studio"
              onClick={() => handleSwitchMode('studio')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xl transition-all ${
                viewMode === 'studio'
                  ? 'bg-gradient-to-r from-cyan-500 to-indigo-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Studio</span>
            </button>

            <button
              type="button"
              data-testid="btn-tab-curtain"
              onClick={() => handleSwitchMode('split_curtain')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xl transition-all ${
                viewMode === 'split_curtain'
                  ? 'bg-gradient-to-r from-cyan-500 to-indigo-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Split className="w-3.5 h-3.5" />
              <span>X-Ray Curtain</span>
            </button>

            <button
              type="button"
              data-testid="btn-tab-split"
              onClick={() => handleSwitchMode('split_diff')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xl transition-all ${
                viewMode === 'split_diff'
                  ? 'bg-gradient-to-r from-cyan-500 to-indigo-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Side-by-Side</span>
            </button>

            <button
              type="button"
              data-testid="btn-tab-clinical"
              onClick={() => handleSwitchMode('clinical_radar')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xl transition-all ${
                viewMode === 'clinical_radar'
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Stethoscope className="w-3.5 h-3.5" />
              <span>Rx Radar</span>
            </button>

            <button
              type="button"
              data-testid="btn-tab-signatures"
              onClick={() => handleSwitchMode('signatures')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xl transition-all ${
                viewMode === 'signatures'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Scale className="w-3.5 h-3.5" />
              <span>Signatures</span>
            </button>
          </div>
        )}

        <div className="flex items-center gap-2 sm:gap-2.5">
          {/* Darkroom Studio Toggle */}
          {document && (
            <button
              type="button"
              data-testid="btn-toggle-darkroom"
              onClick={() => setIsDarkroomOpen((prev) => !prev)}
              className={`p-2 rounded-2xl border transition-colors ${
                isDarkroomOpen
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                  : 'bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border-white/10'
              }`}
              title="Darkroom Image Adjustments"
            >
              <SlidersHorizontal className="w-4 h-4" />
            </button>
          )}

          {/* Diagnostics HUD Button (⌘D) */}
          {document && (
            <button
              type="button"
              data-testid="btn-open-diagnostics"
              onClick={() => setIsDiagnosticsOpen(true)}
              className="p-2 rounded-2xl bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-white/10 shadow-sm transition-colors"
              title="Neural Diagnostics HUD (⌘D)"
            >
              <Activity className="w-4 h-4 text-emerald-400" />
            </button>
          )}

          {/* Quick Command Palette Button (⌘K) */}
          <button
            type="button"
            onClick={() => setIsCommandPaletteOpen(true)}
            data-testid="btn-open-command-palette"
            className="flex items-center gap-2 px-3 py-1.5 rounded-2xl bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-white/10 shadow-sm transition-all group"
            title="Command Palette (⌘K)"
          >
            <Search className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden md:inline text-xs font-medium">Search</span>
            <kbd className="hidden sm:inline px-1.5 py-0.5 rounded-md bg-slate-800 text-[10px] font-mono text-slate-400 border border-slate-700/60 shadow-sm">
              ⌘K
            </kbd>
          </button>

          {/* Shortcuts Guide Button (?) */}
          <button
            type="button"
            onClick={() => setIsShortcutsOpen(true)}
            data-testid="btn-open-shortcuts"
            className="p-2 rounded-2xl bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-white/10 shadow-sm transition-colors"
            title="Keyboard Shortcuts (?)"
          >
            <Keyboard className="w-4 h-4" />
          </button>

          {document && (
            <button
              type="button"
              data-testid="btn-load-new-document"
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-2xl text-xs font-semibold bg-slate-800/90 text-slate-200 hover:bg-slate-700 hover:text-white border border-white/10 hover:border-cyan-500/40 shadow-sm transition-all duration-200"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Load New</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Workspace Area */}
      <main className="flex-1 overflow-hidden p-3 sm:p-4 flex flex-col gap-3">
        {!document ? (
          /* Empty / Ingestion Hero State */
          <div className="flex-1 overflow-y-auto flex flex-col items-center justify-center p-2 sm:p-4 max-w-5xl mx-auto w-full space-y-6">
            {/* Hero Header Banner */}
            <div className="relative w-full rounded-3xl overflow-hidden border border-white/10 bg-gradient-to-b from-slate-900/80 via-slate-900/40 to-slate-950/80 shadow-2xl p-6 sm:p-8 text-center space-y-3.5 backdrop-blur-2xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-medium mb-1 shadow-sm shadow-cyan-500/20">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                <span>Next-Generation Vision-Encoder-Decoder Intelligence</span>
              </div>
              <h2 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
                Read Cursive Notes, Signatures, Archives & Prescriptions with <span className="bg-gradient-to-r from-cyan-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">Apple Silicon AI</span>
              </h2>
              <p className="text-sm text-slate-400 max-w-2xl mx-auto leading-relaxed">
                Fine-tuned on over <strong className="text-slate-200 font-semibold">298,000 real handwriting samples</strong> with double-buffered zero-latency prefetching and RxNorm Trie dictionary rescoring.
              </p>

              {/* Hardware Spec Badges */}
              <div className="flex flex-wrap items-center justify-center gap-2 pt-2 text-[11px] font-mono">
                <span className="px-3 py-1 rounded-xl bg-slate-800/80 border border-slate-700/60 text-slate-300 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-cyan-400" /> Metal GPU MPS Acceleration
                </span>
                <span className="px-3 py-1 rounded-xl bg-slate-800/80 border border-slate-700/60 text-slate-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> SDPA Scaled Dot-Product
                </span>
                <span className="px-3 py-1 rounded-xl bg-slate-800/80 border border-slate-700/60 text-slate-300 flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> 100% Real Handwriting Data
                </span>
              </div>
            </div>

            {/* Dropzone Upload Component */}
            <Dropzone
              onFileAccepted={handleFileAccepted}
              isLoading={isProcessing}
              uploadProgress={uploadProgress}
              processingStage={processingStage}
              onSelectSample={(sampleId) => {
                const preset = SAMPLE_PRESETS[sampleId] || (sampleId === 'sample_legal_contract' ? SAMPLE_LEGAL_CONTRACT : null);
                if (preset) {
                  cleanupObjectURL();
                  setDocument(structuredClone(preset));
                }
              }}
            />

            {error && (
              <div className="w-full p-4 rounded-2xl bg-rose-950/60 border border-rose-800 text-rose-300 text-sm shadow-lg shadow-rose-950/50 flex items-center gap-3">
                <ShieldCheck className="w-5 h-5 text-rose-400 flex-shrink-0" />
                <div>
                  <strong className="font-semibold text-rose-200">Error:</strong> {error}
                </div>
              </div>
            )}

            {/* Preset Sample Gallery */}
            <div className="w-full pt-2">
              <SampleGallery onSelectSample={handleSelectSample} />
            </div>
          </div>
        ) : (
          /* Active Document Loaded State */
          <div className="flex-1 flex flex-col gap-3 overflow-hidden">
            {/* Top KPI Metrics Bar */}
            <MetricsSummary
              document={document}
              onStartSpeedReview={() => {
                const speedTabBtn = window.document.querySelector('[data-testid="tab-speed-review"]') as HTMLButtonElement;
                speedTabBtn?.click();
              }}
            />

            {/* Darkroom Studio Toolbar (Collapsible) */}
            {isDarkroomOpen && (
              <DarkroomToolbar
                settings={darkroom}
                onChange={(s) => setDarkroom(s)}
              />
            )}

            {/* View Mode Router */}
            {viewMode === 'studio' ? (
              /* Studio Mode: Split Screen Workspace (Left: DocumentViewer, Right: InlineEditor) */
              <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-0 overflow-hidden">
                {/* Left Column: Interactive Document Viewer (7 cols) */}
                <div className="lg:col-span-7 h-full flex flex-col min-h-0 overflow-hidden rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl shadow-2xl">
                  {activePage && (
                    <DocumentViewer
                      page={activePage}
                      allPages={document.pages}
                      activePageIndex={activePageIndex}
                      onPageChange={(idx) => setActivePageIndex(idx)}
                      selectedLineId={selectedLineId}
                      selectedWordId={selectedWordId}
                      hoveredLineId={hoveredLineId}
                      hoveredWordId={hoveredWordId}
                      onSelectLine={(lineId) => setSelectedLineId(lineId)}
                      onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                      onHoverLine={(lineId) => setHoveredLineId(lineId)}
                      onHoverWord={(wordId) => setHoveredWordId(wordId)}
                      isLoupeActive={isLoupeActive}
                      onToggleLoupe={() => setIsLoupeActive((prev) => !prev)}
                      isInverted={darkroom.inverted}
                      onToggleInvert={() => setDarkroom((prev) => ({ ...prev, inverted: !prev.inverted }))}
                      contrastBoost={darkroom.contrast}
                    />
                  )}
                </div>

                {/* Right Column: Inline Text Editor & Clinical Intelligence (5 cols) */}
                <div className="lg:col-span-5 h-full flex flex-col gap-2 min-h-0 overflow-hidden">
                  <div className="flex-1 min-h-0 overflow-hidden rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl shadow-2xl flex flex-col">
                    {activePage && (
                      <InlineEditor
                        page={activePage}
                        selectedLineId={selectedLineId}
                        selectedWordId={selectedWordId}
                        onSelectLine={(lineId) => setSelectedLineId(lineId)}
                        onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                        onHoverLine={(lineId) => setHoveredLineId(lineId)}
                        onHoverWord={(wordId) => setHoveredWordId(wordId)}
                        onLineChange={(lineId, newText) => updateLineText(lineId, newText)}
                        onWordChange={(lineId, wordId, newText) => updateWordText(lineId, wordId, newText)}
                        onUndo={undo}
                        onRedo={redo}
                        onRevertAll={revertAll}
                        canUndo={canUndo}
                        canRedo={canRedo}
                      />
                    )}
                  </div>

                  {/* Bottom Export Toolbar */}
                  <ExportToolbar document={document} />
                </div>
              </div>
            ) : viewMode === 'split_curtain' ? (
              /* X-Ray / Slider Curtain Mode */
              <div className="flex-1 min-h-0 overflow-hidden">
                {activePage && <SplitCurtain page={activePage} />}
              </div>
            ) : viewMode === 'split_diff' ? (
              /* Split Diff Mode: Side-by-side original scan vs clean typeset sheet */
              <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-3 min-h-0 overflow-hidden">
                <div className="h-full rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl overflow-hidden flex flex-col">
                  <div className="px-4 py-2.5 bg-slate-950/60 border-b border-white/10 text-xs font-bold text-slate-300 flex items-center justify-between">
                    <span className="flex items-center gap-1.5">
                      <FileText className="w-4 h-4 text-cyan-400" />
                      <span>Original Handwritten Scan</span>
                    </span>
                    <span className="font-mono text-slate-500">Page {activePageIndex + 1}</span>
                  </div>
                  <div className="flex-1 overflow-hidden p-2 flex items-center justify-center bg-slate-950">
                    {activePage?.image_url ? (
                      <img
                        src={activePage.image_url}
                        alt="Original scan"
                        className="max-h-full max-w-full object-contain rounded-lg shadow-xl"
                      />
                    ) : (
                      <div className="text-slate-500 font-mono text-xs">No scan image preview</div>
                    )}
                  </div>
                </div>

                <div className="h-full rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl overflow-hidden flex flex-col">
                  <div className="px-4 py-2.5 bg-slate-950/60 border-b border-white/10 text-xs font-bold text-slate-300 flex items-center justify-between">
                    <span className="flex items-center gap-1.5">
                      <Sparkles className="w-4 h-4 text-indigo-400" />
                      <span>Clean Digital Typeset</span>
                    </span>
                    <span className="text-emerald-400 font-mono font-semibold">
                      {((activePage?.mean_confidence || 0.95) * 100).toFixed(0)}% Conf
                    </span>
                  </div>
                  <div className="flex-1 overflow-y-auto p-6 space-y-4 font-sans text-slate-200 text-sm leading-relaxed bg-slate-900/30">
                    {activePage?.lines.map((line, idx) => (
                      <div
                        key={line.line_id}
                        className="p-3 rounded-xl bg-slate-950/60 border border-white/5 hover:border-cyan-500/30 transition-colors flex items-start gap-3"
                      >
                        <span className="text-xs font-mono text-slate-500 pt-0.5">L{idx + 1}</span>
                        <p className="flex-1 font-medium text-white">{line.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : viewMode === 'signatures' ? (
              /* Signatures & Endorsement Verification Mode */
              <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-0 overflow-hidden">
                <div className="lg:col-span-6 h-full rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl overflow-hidden flex flex-col">
                  {activePage && (
                    <DocumentViewer
                      page={activePage}
                      allPages={document.pages}
                      activePageIndex={activePageIndex}
                      onPageChange={(idx) => setActivePageIndex(idx)}
                      selectedLineId={selectedLineId}
                      selectedWordId={selectedWordId}
                      hoveredLineId={hoveredLineId}
                      hoveredWordId={hoveredWordId}
                      onSelectLine={(lineId) => setSelectedLineId(lineId)}
                      onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                      onHoverLine={(lineId) => setHoveredLineId(lineId)}
                      onHoverWord={(wordId) => setHoveredWordId(wordId)}
                    />
                  )}
                </div>

                <div className="lg:col-span-6 h-full flex flex-col gap-3 min-h-0 overflow-y-auto">
                  {activePage && <SignatureInspector page={activePage} />}

                  <div className="flex-1 min-h-0 overflow-hidden rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl shadow-2xl flex flex-col">
                    {activePage && (
                      <InlineEditor
                        page={activePage}
                        selectedLineId={selectedLineId}
                        selectedWordId={selectedWordId}
                        onSelectLine={(lineId) => setSelectedLineId(lineId)}
                        onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                        onHoverLine={(lineId) => setHoveredLineId(lineId)}
                        onHoverWord={(wordId) => setHoveredWordId(wordId)}
                        onLineChange={(lineId, newText) => updateLineText(lineId, newText)}
                        onWordChange={(lineId, wordId, newText) => updateWordText(lineId, wordId, newText)}
                        onUndo={undo}
                        onRedo={redo}
                        onRevertAll={revertAll}
                        canUndo={canUndo}
                        canRedo={canRedo}
                      />
                    )}
                  </div>

                  <ExportToolbar document={document} />
                </div>
              </div>
            ) : (
              /* Clinical Radar Mode: Deep Clinical Prescription Insights */
              <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-0 overflow-hidden">
                <div className="lg:col-span-6 h-full rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl overflow-hidden flex flex-col">
                  {activePage && (
                    <DocumentViewer
                      page={activePage}
                      allPages={document.pages}
                      activePageIndex={activePageIndex}
                      onPageChange={(idx) => setActivePageIndex(idx)}
                      selectedLineId={selectedLineId}
                      selectedWordId={selectedWordId}
                      hoveredLineId={hoveredLineId}
                      hoveredWordId={hoveredWordId}
                      onSelectLine={(lineId) => setSelectedLineId(lineId)}
                      onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                      onHoverLine={(lineId) => setHoveredLineId(lineId)}
                      onHoverWord={(wordId) => setHoveredWordId(wordId)}
                    />
                  )}
                </div>

                <div className="lg:col-span-6 h-full flex flex-col gap-3 min-h-0 overflow-y-auto">
                  {activePage && (
                    <MedicalEntitiesCard
                      page={activePage}
                      onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                    />
                  )}

                  <div className="flex-1 min-h-0 overflow-hidden rounded-2xl border border-white/10 bg-slate-900/50 backdrop-blur-xl shadow-2xl flex flex-col">
                    {activePage && (
                      <InlineEditor
                        page={activePage}
                        selectedLineId={selectedLineId}
                        selectedWordId={selectedWordId}
                        onSelectLine={(lineId) => setSelectedLineId(lineId)}
                        onSelectWord={(wordId, parentLineId) => setSelectedWordId(wordId, parentLineId)}
                        onHoverLine={(lineId) => setHoveredLineId(lineId)}
                        onHoverWord={(wordId) => setHoveredWordId(wordId)}
                        onLineChange={(lineId, newText) => updateLineText(lineId, newText)}
                        onWordChange={(lineId, wordId, newText) => updateWordText(lineId, wordId, newText)}
                        onUndo={undo}
                        onRedo={redo}
                        onRevertAll={revertAll}
                        canUndo={canUndo}
                        canRedo={canRedo}
                      />
                    )}
                  </div>

                  <ExportToolbar document={document} />
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Global Command Palette Modal (⌘K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        document={document}
        onSelectSample={handleSelectSample}
        onZoomIn={() => {
          const btn = window.document.querySelector('[data-testid="btn-zoom-in"]') as HTMLButtonElement;
          btn?.click();
        }}
        onZoomOut={() => {
          const btn = window.document.querySelector('[data-testid="btn-zoom-out"]') as HTMLButtonElement;
          btn?.click();
        }}
        onZoomFit={() => {
          const btn = window.document.querySelector('[data-testid="btn-zoom-fit"]') as HTMLButtonElement;
          btn?.click();
        }}
        onRotate={() => {
          const btn = window.document.querySelector('[data-testid="btn-rotate"]') as HTMLButtonElement;
          btn?.click();
        }}
        onToggleHeatmap={() => {
          const btn = window.document.querySelector('[data-testid="btn-toggle-heatmap"]') as HTMLButtonElement;
          btn?.click();
        }}
        onToggleBBoxes={() => {
          const btn = window.document.querySelector('[data-testid="btn-toggle-bboxes"]') as HTMLButtonElement;
          btn?.click();
        }}
        onToggleLoupe={() => setIsLoupeActive((prev) => !prev)}
        onToggleInvert={() => setDarkroom((prev) => ({ ...prev, inverted: !prev.inverted }))}
        onStartSpeedReview={() => {
          const btn = window.document.querySelector('[data-testid="tab-speed-review"]') as HTMLButtonElement;
          btn?.click();
        }}
        onExportTxt={handleExportTxt}
        onExportJson={handleExportJson}
        onExportCsv={handleExportCsv}
        onCopyText={handleCopyText}
      />

      {/* Keyboard Shortcuts Sheet Modal (?) */}
      <ShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />

      {/* Real-time Diagnostics HUD Modal (⌘D) */}
      <DiagnosticsHUD
        document={document}
        isOpen={isDiagnosticsOpen}
        onClose={() => setIsDiagnosticsOpen(false)}
      />
    </div>
  );
}
