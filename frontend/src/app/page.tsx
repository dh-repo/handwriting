'use client';

import { LocalDocuments } from '../components/LocalDocuments';
import { rememberOriginal, saveDocument } from '../lib/documentStore';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  PenTool,
  Sparkles,
  RotateCcw,
  Copy,
  Check,
  FileDown,
  Code,
  FileText,
  Layers,
  ChevronDown,
  CheckCircle2,
  FolderOpen,
  Loader2,
} from 'lucide-react';
import { useDocumentContext } from '../context/DocumentContext';
import { Dropzone } from '../components/Dropzone';
import { DocumentViewer } from '../components/DocumentViewer';
import { InlineEditor } from '../components/InlineEditor';
import { SignatureInspector } from '../components/SignatureInspector';
import { DarkroomToolbar, DarkroomSettings } from '../components/DarkroomToolbar';
import { SplitCurtain } from '../components/SplitCurtain';
import { FullWindowDropOverlay } from '../components/FullWindowDropOverlay';
import { StagingQueue } from '../components/StagingQueue';
import { ZeroFrictionSampleCards } from '../components/ZeroFrictionSampleCards';
import { DocumentOCRResult } from '../types/ocr';
import {
  BatchFileItem,
  BatchConfiguration,
} from '../types/batch';
import { findSignatureCandidates } from '../lib/signatureCandidates';
import { apiClient } from '../lib/apiClient';
import {
  exportDocumentAsTxt,
  exportDocumentAsJson,
  exportDocumentAsMarkdown,
  exportBatchDocumentsAsMarkdown,
  exportBatchDocumentsAsTxt,
  exportBatchDocumentsAsJson,
  openSearchablePdfPrint,
  copyTextToClipboard,
  downloadFile,
} from '../lib/exportUtils';

export default function WorkspacePage() {
  const {
    document,
    activePageIndex,
    activePage,
    setDocument,
    appendStreamedLine,
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
    setSignatureDecision,
    revertAll,
    undo,
    redo,
    canUndo,
    canRedo,
  } = useDocumentContext();

  const activeObjectUrlsRef = useRef<string[]>([]);
  const [copied, setCopied] = useState(false);
  const [isLoupeActive, setIsLoupeActive] = useState(false);
  const [viewMode, setViewMode] = useState<'canvas' | 'curtain'>('canvas');
  const [darkroomSettings, setDarkroomSettings] = useState<DarkroomSettings>({
    contrast: 1.0,
    brightness: 1.0,
    grayscale: false,
    inverted: false,
  });

  // Batch Ingestion & Queue State
  const [stagedFiles, setStagedFiles] = useState<BatchFileItem[]>([]);
  const [batchConfig, setBatchConfig] = useState<BatchConfiguration>({
    outputMode: 'single',
    modelBias: 'general',
    format: 'markdown',
  });
  const [batchOverallProgress, setBatchOverallProgress] = useState<number>(0);
  const [currentBatchIndex, setCurrentBatchIndex] = useState<number>(0);
  const [batchStageText, setBatchStageText] = useState<string>('');
  const [completedBatchDocuments, setCompletedBatchDocuments] = useState<DocumentOCRResult[]>([]);
  const [activeBatchDocIndex, setActiveBatchDocIndex] = useState<number>(0);
  const [engineMode, setEngineMode] = useState<'turbo' | 'trocr'>('trocr');
  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const exportDropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (exportDropdownRef.current && !exportDropdownRef.current.contains(e.target as Node)) {
        setIsExportMenuOpen(false);
      }
    };
    if (isExportMenuOpen) {
      window.addEventListener('mousedown', handleOutsideClick);
    }
    return () => {
      window.removeEventListener('mousedown', handleOutsideClick);
    };
  }, [isExportMenuOpen]);

  const cleanupObjectURLs = useCallback(() => {
    activeObjectUrlsRef.current.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // ignore
      }
    });
    activeObjectUrlsRef.current = [];
  }, []);

  useEffect(() => {
    return () => {
      cleanupObjectURLs();
    };
  }, [cleanupObjectURLs]);

  // Update a single item in the staged files queue
  const updateStagedItem = useCallback((id: string, partial: Partial<BatchFileItem>) => {
    setStagedFiles((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...partial } : item))
    );
  }, []);

  // Single file direct recognition (fast-path)
  const handleSingleFileAccepted = useCallback(
    async (file: File) => {
      const originalKey = await rememberOriginal(file).catch(() => { setError("Original could not be saved locally; export your result"); return undefined; });
      setIsProcessing(true);
      setUploadProgress(15);
      setProcessingStage('Uploading document & preparing preprocessor...');
      setError(null);


      let previewUrl: string | undefined;
      if (file.type.startsWith('image/')) {
        previewUrl = URL.createObjectURL(file);
        activeObjectUrlsRef.current.push(previewUrl);

        // Show initial document preview immediately so viewer is not blank
        setDocument({
          document_id: `pending_${originalKey || Date.now()}`,
          original_key: originalKey, incomplete: true,
          filename: file.name,
          total_pages: 1,
          pages: [
            {
              page_number: 1,
              width: 1000,
              height: 1400,
              full_text: '',
              mean_confidence: null,
              lines: [],
              image_url: previewUrl,
            },
          ],
          processing_time_ms: 0,
        });
      }

      const startTime = Date.now();
      let totalExpectedLines = 0;
      let decodedCount = 0;

      const progressInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        if (decodedCount === 0) {
          if (engineMode === 'turbo') {
            if (elapsed < 2) {
              setUploadProgress(30);
              setProcessingStage('⚡ Segmenting document & querying Turbo VLM...');
            } else {
              setUploadProgress(65);
              setProcessingStage('⚡ Turbo VLM analyzing cursive handwriting...');
            }
          } else {
            if (elapsed < 3) {
              setUploadProgress(20);
              setProcessingStage('Preprocessing, deskewing & segmenting line crops...');
            } else if (elapsed < 6) {
              setUploadProgress(28);
              setProcessingStage('Vision transformer extracting handwriting stroke tokens...');
            } else {
              setUploadProgress(35);
              setProcessingStage(`Neural decoding lines (${elapsed}s elapsed)...`);
            }
          }
        }
      }, 1000);

      try {
        const result: DocumentOCRResult = await apiClient.recognizeFileStream(
          file,
          {
            beam_width: 4,
            rescore: false,
            adaptive: true,
            processing_mode: engineMode === 'turbo' ? 'cloud' : 'local',
            model_type: batchConfig.modelBias,
          },
          {
            onMetadata: (meta) => {
              setDocument({ document_id: meta.document_id, filename: meta.filename, total_pages: meta.total_pages,
                original_key: originalKey, incomplete: true, processing_time_ms: 0,
                processing_location: engineMode === 'turbo' ? 'cloud' : 'local',
                pages: meta.pages.map((p: any) => ({ ...p, full_text: '', mean_confidence: null, lines: [] })) });
              if (meta.pages && meta.pages[0]) {
                totalExpectedLines = meta.pages[0].total_lines || 0;
                setProcessingStage(`Segmented into ${totalExpectedLines} lines. Decoding strokes...`);
                setUploadProgress(30);
              }
            },
            onLine: (line, pageNum) => {
              decodedCount++;
              appendStreamedLine(line, pageNum);
              const total = Math.max(1, totalExpectedLines || 13);
              const pct = Math.min(95, 30 + Math.floor((decodedCount / total) * 65));
              setUploadProgress(pct);
              const snippet = line.text.length > 28 ? `${line.text.slice(0, 28)}...` : line.text;
              setProcessingStage(`Decoded line ${decodedCount}/${total}: "${snippet}"`);
            },
            onComplete: (doc) => {
              doc.original_key = originalKey;
              if (previewUrl && doc.pages[0]) {
                doc.pages[0].image_url = previewUrl;
              }
              setDocument(doc);
              setCompletedBatchDocuments([doc]);
              setUploadProgress(100);
              setProcessingStage('Complete');
            },
            onError: (errMsg) => {
              console.warn('[RecognizeStream Error]', errMsg);
            },
          }
        );

        clearInterval(progressInterval);

        if (previewUrl && result.pages[0]) {
          result.pages[0].image_url = previewUrl;
        }
        setDocument(result);
        setCompletedBatchDocuments([result]);
        setUploadProgress(100);
        setProcessingStage('Complete');
      } catch (err: unknown) {
        clearInterval(progressInterval);
        const msg = err instanceof Error ? err.message : 'Failed to process document';
        setError(msg);
      } finally {
        clearInterval(progressInterval);
        setIsProcessing(false);
        setEngineMode("trocr");
        setUploadProgress(0);
      }
    },
    [
      engineMode,
      batchConfig.modelBias,
      cleanupObjectURLs,
      appendStreamedLine,
      setDocument,
      setError,
      setIsProcessing,
      setProcessingStage,
      setUploadProgress,
    ]
  );

  // Ingestion handler for multiple files
  const handleFilesAccepted = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;

      // If exactly 1 file and queue is currently empty, execute immediate fast-path
      if (files.length === 1 && stagedFiles.length === 0 && !document) {
        handleSingleFileAccepted(files[0]);
        return;
      }

      // Multiple files or queue already exists -> stage into batch queue
      const newItems: BatchFileItem[] = files.map((file) => {
        let previewUrl: string | undefined;
        if (file.type.startsWith('image/')) {
          try {
            previewUrl = URL.createObjectURL(file);
            activeObjectUrlsRef.current.push(previewUrl);
          } catch {
            // ignore
          }
        }
        return {
          id: `file_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          file,
          name: file.name,
          size: file.size,
          type: file.type,
          previewUrl,
          pageCount: file.type === 'application/pdf' ? 3 : 1,
          status: 'staged',
          progress: 0,
        };
      });

      setStagedFiles((prev) => [...prev, ...newItems]);
    },
    [stagedFiles.length, document, handleSingleFileAccepted]
  );

  // Start Batch Transcription over all staged files
  const handleStartBatchExecution = async () => {
    if (stagedFiles.length === 0) return;

    setIsProcessing(true);
    setBatchOverallProgress(5);
    setCompletedBatchDocuments([]);
    setError(null);

    const completedDocs: DocumentOCRResult[] = [];

    for (let i = 0; i < stagedFiles.length; i++) {
      setCurrentBatchIndex(i);
      const currentItem = stagedFiles[i];
      const originalKey = await rememberOriginal(currentItem.file).catch(() => { setError("Original could not be saved locally"); return undefined; });

      updateStagedItem(currentItem.id, {
        status: 'processing',
        stageText: 'Transcribing strokes...',
        progress: 10,
      });
      setBatchStageText(
        `Processing file ${i + 1} of ${stagedFiles.length}: "${currentItem.name}"...`
      );

      // If first document, show preview in workspace right away
      if (i === 0 && currentItem.previewUrl) {
        setDocument({
          document_id: `pending_${originalKey || Date.now()}`,
          original_key: originalKey,
          incomplete: true,
          filename: currentItem.name,
          total_pages: 1,
          pages: [
            {
              page_number: 1,
              width: 1000,
              height: 1400,
              full_text: '',
              mean_confidence: null,
              lines: [],
              image_url: currentItem.previewUrl,
            },
          ],
          processing_time_ms: 0,
        });
      }

      try {
        let totalLines = 12;
        let decoded = 0;

        const result: DocumentOCRResult = await apiClient.recognizeFileStream(
          currentItem.file,
          {
            model_type: batchConfig.modelBias,
            beam_width: 4,
            rescore: false,
            adaptive: true,
            processing_mode: engineMode === 'turbo' ? 'cloud' : 'local',
          },
          {
            onMetadata: (meta) => {
              setDocument({ document_id: meta.document_id, filename: meta.filename, total_pages: meta.total_pages,
                original_key: originalKey, incomplete: true, processing_time_ms: 0,
                processing_location: engineMode === 'turbo' ? 'cloud' : 'local',
                pages: meta.pages.map((p: any) => ({ ...p, full_text: '', mean_confidence: null, lines: [] })) });
              if (meta.pages && meta.pages[0]) {
                totalLines = meta.pages[0].total_lines || 12;
                updateStagedItem(currentItem.id, {
                  totalLinesCount: totalLines,
                  stageText: `Segmented ${totalLines} lines. Decoding...`,
                });
              }
            },
            onLine: (line, pageNum) => {
              decoded++;
              if (i === 0) {
                appendStreamedLine(line, pageNum);
              }
              const filePct = Math.min(95, Math.round((decoded / totalLines) * 90));
              updateStagedItem(currentItem.id, {
                decodedLinesCount: decoded,
                progress: filePct,
                stageText: `Decoded ${decoded}/${totalLines} lines`,
              });

              const batchPct = Math.round(((i + decoded / totalLines) / stagedFiles.length) * 95);
              setBatchOverallProgress(batchPct);
              setBatchStageText(
                `Processing file ${i + 1} of ${stagedFiles.length} • Decoded line ${decoded}/${totalLines}`
              );
            },
            onComplete: (doc) => {
              doc.original_key = originalKey;
              if (currentItem.previewUrl && doc.pages[0]) {
                doc.pages[0].image_url = currentItem.previewUrl;
              }
            },
            onError: (errMsg) => {
              console.warn('[Batch RecognizeStream Error]', errMsg);
            },
          }
        );

        if (currentItem.previewUrl && result.pages[0]) {
          result.pages[0].image_url = currentItem.previewUrl;
        }

        result.original_key = originalKey;
        await saveDocument(result).catch(() => setError("Local save failed — export this document"));
        completedDocs.push(result);
        updateStagedItem(currentItem.id, {
          status: 'completed',
          progress: 100,
          result,
          stageText: 'Completed',
        });

        // If this is the first file, set document so split-pane verification is ready immediately
        if (i === 0) {
          setDocument(result);
          setActiveBatchDocIndex(0);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Recognition error';
        updateStagedItem(currentItem.id, {
          status: 'error',
          error: msg,
          stageText: 'Failed',
        });
      }
    }

    setCompletedBatchDocuments(completedDocs);
    setBatchOverallProgress(100);
    setBatchStageText(`Batch complete • ${completedDocs.length} documents transcribed.`);
    setIsProcessing(false);
        setEngineMode("trocr");
  };

  const handleReset = () => {
    cleanupObjectURLs();
    setDocument(null);
    setStagedFiles([]);
    setCompletedBatchDocuments([]);
    setActiveBatchDocIndex(0);
    setError(null);
    setIsProcessing(false);
        setEngineMode("trocr");
  };

  const handleCopyText = async () => {
    if (!document) return;
    const fullText = document.pages
      .map((p) => p.lines.map((l) => l.text).join('\n'))
      .join('\n\n');
    const success = await copyTextToClipboard(fullText);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Export handlers
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

  const handleExportMarkdown = () => {
    if (!document) return;
    const content = exportDocumentAsMarkdown(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_transcription.md`;
    downloadFile(content, filename, 'text/markdown;charset=utf-8');
  };

  const handleExportSearchablePdf = () => {
    if (!document) return;
    openSearchablePdfPrint(document);
  };

  const handleJumpToNextAmbiguity = () => {
    if (!document || !activePage) return;
    const ambiguousTokens: { wordId: string; lineId: string }[] = [];
    for (const line of activePage.lines) {
      for (const word of line.words || []) {
        if ((word.confidence == null || (word.confidence ?? 0) < 0.85)) {
          ambiguousTokens.push({ wordId: word.word_id, lineId: line.line_id });
        }
      }
    }
    if (ambiguousTokens.length === 0) return;

    const currentIdx = ambiguousTokens.findIndex((t) => t.wordId === selectedWordId);
    const nextIdx = (currentIdx + 1) % ambiguousTokens.length;
    const next = ambiguousTokens[nextIdx];
    setSelectedWordId(next.wordId, next.lineId);
    setSelectedLineId(next.lineId);
  };

  const handleExportBatchCombined = () => {
    if (completedBatchDocuments.length === 0) return;

    if (batchConfig.format === 'markdown') {
      const content = exportBatchDocumentsAsMarkdown(completedBatchDocuments.map(d => d.document_id === document?.document_id ? document : d));
      downloadFile(content, 'batch_transcriptions.md', 'text/markdown;charset=utf-8');
    } else if (batchConfig.format === 'json') {
      const content = exportBatchDocumentsAsJson(completedBatchDocuments.map(d => d.document_id === document?.document_id ? document : d));
      downloadFile(content, 'batch_ocr_results.json', 'application/json');
    } else {
      const content = exportBatchDocumentsAsTxt(completedBatchDocuments.map(d => d.document_id === document?.document_id ? document : d));
      downloadFile(content, 'batch_transcriptions.txt', 'text/plain;charset=utf-8');
    }
  };

  // Compute metrics for active document
  const totalWords = document
    ? document.pages.reduce(
        (acc, p) =>
          acc +
          p.lines.reduce(
            (lAcc, l) => lAcc + (l.words?.length || l.text.split(' ').filter(Boolean).length),
            0
          ),
        0
      )
    : 0;

  const totalLines = document
    ? document.pages.reduce((acc, p) => acc + p.lines.length, 0)
    : 0;

  const meanConfidence = document?.pages.some(p => p.mean_confidence == null) ? null : document
    ? Math.round(
        (document.pages.reduce((acc, p) => acc + (p.mean_confidence ?? 0), 0) /
          Math.max(1, document.pages.length)) *
          100
      )
    : 0;

  const lowConfidenceCount = document
    ? document.pages.reduce(
        (acc, p) =>
          acc +
          p.lines.reduce(
            (lAcc, l) => lAcc + l.words.filter((w) => (w.confidence == null || (w.confidence ?? 0) < 0.85)).length,
            0
          ),
        0
      )
    : 0;

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200 antialiased">
      {/* Full-Viewport Drop Target Overlay */}
      <FullWindowDropOverlay onFilesDropped={handleFilesAccepted} disabled={isProcessing} />

      <LocalDocuments />
      {/* Unified Compact Top Navigation Bar (h-14 / 56px) */}
      <header className="sticky top-0 z-40 h-14 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-2xl px-4 sm:px-6 flex items-center justify-between">
        {/* Left Section: Brand & Document Metadata */}
        <div className="flex items-center gap-3 min-w-0 overflow-hidden">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-blue-500 flex items-center justify-center shadow-md shadow-indigo-500/20 border border-white/20 shrink-0">
            <PenTool className="w-4 h-4 text-white" />
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="font-semibold text-zinc-100 tracking-tight text-sm">Handwriting AI</span>
            {!document && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide uppercase bg-blue-500/15 text-blue-400 border border-blue-500/30">
                Neural Engine
              </span>
            )}
          </div>

          {document && (
            <div className="hidden md:flex items-center gap-2.5 text-xs text-zinc-400 pl-2 border-l border-zinc-800 min-w-0">
              {/* Multi-Document Batch Switcher Tabs (If batch processed) */}
              {completedBatchDocuments.length > 1 && (
                <div className="flex items-center gap-1 p-0.5 bg-zinc-900 rounded-lg border border-zinc-800 mr-1">
                  {completedBatchDocuments.map((doc, idx) => (
                    <button
                      key={`${doc.document_id || 'doc'}_${idx}`}
                      type="button"
                      data-testid={`batch-tab-doc-${idx}`}
                      onClick={() => {
                        setActiveBatchDocIndex(idx);
                        setDocument(doc);
                      }}
                      className={`px-2 py-0.5 rounded-md font-medium text-[11px] transition-all ${
                        activeBatchDocIndex === idx
                          ? 'bg-zinc-100 text-zinc-900 font-semibold shadow-sm'
                          : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      Doc {idx + 1}
                    </button>
                  ))}
                </div>
              )}

              <span className="font-medium text-zinc-200 truncate max-w-[160px] lg:max-w-xs">
                {document.filename || 'Document'}
              </span>
              <span className="text-zinc-600 select-none">•</span>
              <span>
                {totalLines} {totalLines === 1 ? 'Line' : 'Lines'}
              </span>
              <span className="text-zinc-600 select-none">•</span>
              <span>{totalWords} Words</span>
              <span className="text-zinc-600 select-none">•</span>
              <span
                className={
                  (meanConfidence ?? -1) >= 90
                    ? 'text-emerald-400 font-semibold'
                    : (meanConfidence ?? -1) >= 75
                    ? 'text-amber-400 font-semibold'
                    : 'text-rose-400 font-semibold'
                }
              >
                {meanConfidence == null ? "Unknown" : `${meanConfidence}%`} Confidence
              </span>

              {lowConfidenceCount > 0 && (
                <>
                  <span className="text-zinc-600 select-none">•</span>
                  <button
                    type="button"
                    onClick={handleJumpToNextAmbiguity}
                    title="Click to jump through low-confidence tokens"
                    data-testid="low-confidence-counter-badge"
                    className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30 hover:border-amber-400 transition-all cursor-pointer shadow-sm group"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse group-hover:scale-125 transition-transform" />
                    <span>{lowConfidenceCount} ambiguous</span>
                  </button>
                </>
              )}

              <span className="text-zinc-600 select-none">•</span>
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold ${
                  document?.engine_used === 'turbo-vlm' || (document?.preprocessing_flags as any)?.engine === 'turbo-vlm'
                    ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                    : 'bg-blue-500/15 text-blue-300 border border-blue-500/30'
                }`}
              >
                {document?.engine_used === 'turbo-vlm' || (document?.preprocessing_flags as any)?.engine === 'turbo-vlm' ? '⚡ Turbo' : '🧠 TrOCR'}
                {document?.processing_time_ms ? ` (${(document.processing_time_ms / 1000).toFixed(1)}s)` : ''}
              </span>
            </div>
          )}
        </div>

        {/* Right Section: Actions & Split Export Dropdown */}
        {document && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopyText}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 transition-all duration-150 shadow-sm"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5 text-zinc-400" />
              )}
              <span className="hidden sm:inline">{copied ? 'Copied' : 'Copy Text'}</span>
            </button>

            {/* Split Export ▾ Dropdown */}
            <div className="relative" ref={exportDropdownRef}>
              <div className="inline-flex rounded-xl shadow-sm border border-zinc-700/80 bg-zinc-800/90 hover:bg-zinc-800 divide-x divide-zinc-700/80 overflow-hidden">
                <button
                  type="button"
                  data-testid="btn-export-searchable-pdf"
                  onClick={handleExportSearchablePdf}
                  title="Export searchable PDF with OCR text layer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-zinc-100 hover:bg-zinc-700/50 transition-colors"
                >
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  <span>Export PDF</span>
                </button>
                <button
                  type="button"
                  onClick={() => setIsExportMenuOpen((prev) => !prev)}
                  title="More export formats"
                  className="px-2 py-1.5 text-zinc-300 hover:text-white hover:bg-zinc-700/50 transition-colors"
                  aria-expanded={isExportMenuOpen}
                >
                  <ChevronDown className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Dropdown Menu */}
              {isExportMenuOpen && (
                <div className="absolute right-0 mt-1.5 w-48 rounded-xl bg-zinc-900 border border-zinc-700/80 shadow-2xl backdrop-blur-xl p-1 z-50 animate-in fade-in slide-in-from-top-1 text-xs">
                  <button
                    type="button"
                    onClick={() => {
                      setIsExportMenuOpen(false);
                      handleExportMarkdown();
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-zinc-300 hover:text-white hover:bg-zinc-800/80 transition-colors text-left"
                  >
                    <FileText className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Export Markdown (.md)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsExportMenuOpen(false);
                      handleExportTxt();
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-zinc-300 hover:text-white hover:bg-zinc-800/80 transition-colors text-left"
                  >
                    <FileDown className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Export Plain Text (.txt)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsExportMenuOpen(false);
                      handleExportJson();
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-zinc-300 hover:text-white hover:bg-zinc-800/80 transition-colors text-left"
                  >
                    <Code className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Export JSON (.json)</span>
                  </button>

                  {completedBatchDocuments.length > 1 && batchConfig.outputMode === 'single' && (
                    <div className="pt-1 mt-1 border-t border-zinc-800">
                      <button
                        type="button"
                        data-testid="btn-export-combined-batch"
                        onClick={() => {
                          setIsExportMenuOpen(false);
                          handleExportBatchCombined();
                        }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-indigo-300 hover:text-white hover:bg-indigo-600/30 transition-colors text-left font-medium"
                      >
                        <Layers className="w-3.5 h-3.5 text-indigo-400" />
                        <span>Export All ({completedBatchDocuments.length})</span>
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            <button
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 transition-all duration-200 shadow-sm"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>New Scan</span>
            </button>
          </div>
        )}
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col p-3 sm:p-5 lg:p-6 max-w-[1700px] w-full mx-auto">
        {!document && stagedFiles.length === 0 ? (
          /* Landing Screen: Expanded Drop Target & Zero-Friction Sample Cards */
          <div className="flex-1 flex flex-col items-center justify-center py-6 sm:py-12 max-w-4xl mx-auto w-full text-center space-y-6 animate-in fade-in">
            <div className="space-y-3">
              <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-medium text-blue-400 bg-blue-500/10 border border-blue-500/20 backdrop-blur-md">
                <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                <span>Private handwriting review</span>
              </div>
              <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.15]">
                Transcribe Handwriting with{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400">
                  Human Review
                </span>
              </h1>
              <p className="text-sm sm:text-base text-zinc-400 max-w-2xl mx-auto font-normal leading-relaxed">
                Transcribe notes and letters, compare the original, and correct the result before exporting.
              </p>

              {/* ReadMe-Inspired Engine Selection Capsule */}
              <div className="pt-1 flex justify-center">
                <div className="inline-flex items-center p-1 bg-zinc-900/90 border border-zinc-800 rounded-full backdrop-blur-xl shadow-lg">
                  <button
                    type="button"
                    onClick={() => { if (window.confirm('Send selected documents to Azure for cloud recognition? Corrections stay local.')) setEngineMode('turbo'); }}
                    data-testid="engine-toggle-turbo"
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
                      engineMode === 'turbo'
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 shadow-sm shadow-amber-500/10'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <span className="text-amber-400 font-bold">⚡</span>
                    <span>Cloud</span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber-500/15 text-amber-300 font-medium">Azure • opt-in</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setEngineMode('trocr')}
                    data-testid="engine-toggle-trocr"
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
                      engineMode === 'trocr'
                        ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30 shadow-sm shadow-blue-500/10'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <span>🧠</span>
                    <span>Neural TrOCR</span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-blue-500/15 text-blue-300 font-medium">On-Device</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Expanded Dropzone Component with Source Parity */}
            <Dropzone
              onFileAccepted={handleSingleFileAccepted}
              onFilesAccepted={handleFilesAccepted}
              onSelectSample={(sample) => {
                setDocument({ ...sample, is_demo: true, engine_used: "demo" });
                setCompletedBatchDocuments([sample]);
              }}
              isLoading={isProcessing}
              uploadProgress={uploadProgress}
              processingStage={processingStage}
            />

            {error && (
              <div className="max-w-md w-full p-4 rounded-2xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs shadow-2xl backdrop-blur-xl animate-in fade-in">
                <p className="font-semibold mb-1">Recognition Error</p>
                <p className="text-rose-400/90">{error}</p>
              </div>
            )}
          </div>
        ) : !document && stagedFiles.length > 0 ? (
          /* Staging Queue View: Handling Multiple Files */
          <div className="flex-1 flex flex-col items-center justify-center py-8 max-w-4xl mx-auto w-full animate-in fade-in">
            <StagingQueue
              items={stagedFiles}
              config={batchConfig}
              onConfigChange={setBatchConfig}
              onRemoveItem={(id) => setStagedFiles((prev) => prev.filter((it) => it.id !== id))}
              onAddMoreFiles={handleFilesAccepted}
              onStartBatch={handleStartBatchExecution}
              onClearQueue={() => setStagedFiles([])}
              isProcessing={isProcessing}
              overallProgress={batchOverallProgress}
              currentProcessingIndex={currentBatchIndex}
              overallStageText={batchStageText}
              onSelectPreviewDoc={(doc) => setDocument(doc)}
            />
          </div>
        ) : (
          /* Workspace Screen: Split-Pane Side-by-Side Verification */
          <div className="flex-1 flex flex-col gap-3 min-h-0 animate-in fade-in">
            {/* Active Workspace Recognition Status Cockpit */}
            {isProcessing && (
              <div
                data-testid="workspace-processing-banner"
                className="w-full bg-zinc-900/90 border border-blue-500/30 backdrop-blur-2xl rounded-2xl px-5 py-3 shadow-2xl flex flex-wrap items-center justify-between gap-3 animate-in fade-in slide-in-from-top-2"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-blue-500/15 border border-blue-500/30 flex items-center justify-center text-blue-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-zinc-100">
                        {engineMode === 'turbo' ? 'Cloud recognition selected' : 'Local recognition selected'}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-300 border border-blue-500/25">
                        Transcribing...
                      </span>
                    </div>
                    <p className="text-xs text-zinc-400 font-mono mt-0.5 max-w-xl truncate">
                      {processingStage}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4 ml-auto">
                  {/* Progress bar */}
                  <div className="w-36 sm:w-48 space-y-1">
                    <div className="flex items-center justify-between text-[11px] font-mono">
                      <span className="text-zinc-500">Pipeline</span>
                      <span className="text-blue-400 font-semibold">{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden border border-zinc-700/50">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-emerald-400 h-full rounded-full transition-all duration-300"
                        style={{ width: `${Math.max(10, Math.min(100, uploadProgress))}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
            {/* Split-Pane Verification Workspace */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0 overflow-hidden">
              {/* Left Column: Canvas Viewport with Floating Glass HUD Dock (7 cols) */}
              <div className="lg:col-span-7 h-full flex flex-col min-h-[500px] overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-950/60 shadow-2xl relative">
                {/* Floating Glass HUD Dock */}
                <div className="absolute top-3.5 left-3.5 right-3.5 z-30 flex items-center justify-between pointer-events-none">
                  {/* Left HUD: View Mode Toggle */}
                  <div className="pointer-events-auto flex items-center gap-1 bg-zinc-900/85 hover:bg-zinc-900 backdrop-blur-xl p-1 rounded-xl border border-zinc-700/60 shadow-xl text-xs">
                    <button
                      type="button"
                      data-testid="toggle-view-canvas"
                      onClick={() => setViewMode('canvas')}
                      className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                        viewMode === 'canvas'
                          ? 'bg-zinc-100 text-zinc-900 font-semibold shadow-sm'
                          : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      Canvas Inspector
                    </button>
                    <button
                      type="button"
                      data-testid="toggle-view-curtain"
                      onClick={() => setViewMode('curtain')}
                      className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                        viewMode === 'curtain'
                          ? 'bg-zinc-100 text-zinc-900 font-semibold shadow-sm'
                          : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      Split Curtain X-Ray
                    </button>
                  </div>

                  {/* Right HUD: Darkroom Controls */}
                  {activePage && (
                    <div className="pointer-events-auto bg-zinc-900/85 hover:bg-zinc-900 backdrop-blur-xl rounded-xl border border-zinc-700/60 shadow-xl p-1">
                      <DarkroomToolbar
                        settings={darkroomSettings}
                        onChange={setDarkroomSettings}
                        className="border-0 bg-transparent p-0"
                      />
                    </div>
                  )}
                </div>

                {/* Full-bleed Canvas Viewport */}
                <div className="flex-1 w-full h-full min-h-0 overflow-hidden relative">
                  {activePage && viewMode === 'canvas' && (
                    <DocumentViewer
                      page={activePage}
                      allPages={document?.pages || [activePage]}
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
                      isInverted={darkroomSettings.inverted}
                      onToggleInvert={() =>
                        setDarkroomSettings((prev) => ({ ...prev, inverted: !prev.inverted }))
                      }
                      contrastBoost={darkroomSettings.contrast}
                      signatureLineIds={[]}
                    />
                  )}
                  {activePage && viewMode === 'curtain' && (
                    <SplitCurtain page={activePage} />
                  )}
                </div>
              </div>

              {/* Right Column: Editable Transcription (5 cols) */}
              <div className="lg:col-span-5 h-full flex flex-col min-h-[500px] overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-950/60 shadow-2xl">
                {activePage && (
                  <div className="flex h-full min-h-0 flex-col">
                    <div className="min-h-0 flex-1 overflow-hidden">
                      <InlineEditor
                        page={activePage}
                        documentId={document?.is_demo ? "demo" : document?.document_id}
                        selectedLineId={selectedLineId}
                        selectedWordId={selectedWordId}
                        hoveredLineId={hoveredLineId}
                        hoveredWordId={hoveredWordId}
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
                    </div>

                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
