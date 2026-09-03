'use client';

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
      setIsProcessing(true);
      setUploadProgress(15);
      setProcessingStage('Uploading document & preparing preprocessor...');
      setError(null);

      cleanupObjectURLs();
      let previewUrl: string | undefined;
      if (file.type.startsWith('image/')) {
        previewUrl = URL.createObjectURL(file);
        activeObjectUrlsRef.current.push(previewUrl);

        // Show initial document preview immediately so viewer is not blank
        setDocument({
          document_id: 'doc_streaming',
          filename: file.name,
          total_pages: 1,
          pages: [
            {
              page_number: 1,
              width: 1000,
              height: 1400,
              full_text: '',
              mean_confidence: 1.0,
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
      }, 1000);

      try {
        const result: DocumentOCRResult = await apiClient.recognizeFileStream(
          file,
          {
            beam_width: 4,
            rescore: false,
            adaptive: true,
            model_type: batchConfig.modelBias,
          },
          {
            onMetadata: (meta) => {
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
        setUploadProgress(0);
      }
    },
    [
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
          document_id: 'doc_streaming_batch',
          filename: currentItem.name,
          total_pages: 1,
          pages: [
            {
              page_number: 1,
              width: 1000,
              height: 1400,
              full_text: '',
              mean_confidence: 1.0,
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
          },
          {
            onMetadata: (meta) => {
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
  };

  const handleReset = () => {
    cleanupObjectURLs();
    setDocument(null);
    setStagedFiles([]);
    setCompletedBatchDocuments([]);
    setActiveBatchDocIndex(0);
    setError(null);
    setIsProcessing(false);
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
        if (word.confidence < 0.85) {
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
      const content = exportBatchDocumentsAsMarkdown(completedBatchDocuments);
      downloadFile(content, 'batch_transcriptions.md', 'text/markdown;charset=utf-8');
    } else if (batchConfig.format === 'json') {
      const content = exportBatchDocumentsAsJson(completedBatchDocuments);
      downloadFile(content, 'batch_ocr_results.json', 'application/json');
    } else {
      const content = exportBatchDocumentsAsTxt(completedBatchDocuments);
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

  const meanConfidence = document
    ? Math.round(
        (document.pages.reduce((acc, p) => acc + (p.mean_confidence || 0.9), 0) /
          Math.max(1, document.pages.length)) *
          100
      )
    : 0;

  const lowConfidenceCount = document
    ? document.pages.reduce(
        (acc, p) =>
          acc +
          p.lines.reduce(
            (lAcc, l) => lAcc + l.words.filter((w) => w.confidence < 0.85).length,
            0
          ),
        0
      )
    : 0;

  return (
    <div className="min-h-screen bg-[#07090E] text-slate-100 flex flex-col font-sans selection:bg-blue-500/30 selection:text-blue-200 antialiased">
      {/* Full-Viewport Drop Target Overlay */}
      <FullWindowDropOverlay onFilesDropped={handleFilesAccepted} disabled={isProcessing} />

      {/* Apple HIG Top Navigation Bar */}
      <header className="sticky top-0 z-40 h-16 border-b border-white/[0.08] bg-[#07090E]/80 backdrop-blur-2xl px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/25 border border-white/20">
            <PenTool className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-white tracking-tight text-base">Handwriting AI</span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide uppercase bg-blue-500/15 text-blue-400 border border-blue-500/30">
                Neural Engine
              </span>
            </div>
          </div>
        </div>

        {document && (
          <div className="flex items-center gap-3">
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium bg-white/[0.06] hover:bg-white/[0.12] text-slate-300 hover:text-white border border-white/10 hover:border-white/20 transition-all duration-200 shadow-sm"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>New Scan</span>
            </button>
          </div>
        )}
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col p-4 sm:p-6 lg:p-8 max-w-[1600px] w-full mx-auto">
        {!document && stagedFiles.length === 0 ? (
          /* Landing Screen: Expanded Drop Target & Zero-Friction Sample Cards */
          <div className="flex-1 flex flex-col items-center justify-center py-8 sm:py-16 max-w-4xl mx-auto w-full text-center space-y-8 animate-in fade-in">
            <div className="space-y-3">
              <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-medium text-blue-400 bg-blue-500/10 border border-blue-500/20 backdrop-blur-md">
                <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                <span>Deep Vision-Language Intelligence</span>
              </div>
              <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.15]">
                Transcribe Handwriting with{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400">
                  Neural Precision
                </span>
              </h1>
              <p className="text-sm sm:text-base text-slate-400 max-w-2xl mx-auto font-normal leading-relaxed">
                Effortlessly read cursive correspondence, archival notes, signatures, receipts, and multi-page documents.
              </p>
            </div>

            {/* Expanded Dropzone Component with Source Parity */}
            <Dropzone
              onFileAccepted={handleSingleFileAccepted}
              onFilesAccepted={handleFilesAccepted}
              onSelectSample={(sample) => {
                setDocument(sample);
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
          <div className="flex-1 flex flex-col gap-4 min-h-0 animate-in fade-in">
            {/* Top Document Summary & Batch Switcher Bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-2xl shadow-xl">
              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-300">
                {/* Multi-Document Batch Switcher Tabs (If batch processed) */}
                {completedBatchDocuments.length > 1 && (
                  <div className="flex items-center gap-1.5 p-1 bg-white/[0.06] rounded-xl border border-white/10 mr-2">
                    {completedBatchDocuments.map((doc, idx) => (
                      <button
                        key={`${doc.document_id || 'doc'}_${idx}`}
                        type="button"
                        data-testid={`batch-tab-doc-${idx}`}
                        onClick={() => {
                          setActiveBatchDocIndex(idx);
                          setDocument(doc);
                        }}
                        className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                          activeBatchDocIndex === idx
                            ? 'bg-blue-600 text-white shadow-sm font-semibold'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Doc {idx + 1}
                      </button>
                    ))}
                  </div>
                )}

                <span className="font-semibold text-white truncate max-w-xs">
                  {document?.filename || 'Document'}
                </span>
                <span className="text-slate-600">•</span>
                <span>
                  {totalLines} {totalLines === 1 ? 'Line' : 'Lines'}
                </span>
                <span className="text-slate-600">•</span>
                <span>{totalWords} Words</span>
                <span className="text-slate-600">•</span>
                <span
                  className={
                    meanConfidence >= 90
                      ? 'text-emerald-400 font-semibold'
                      : meanConfidence >= 75
                      ? 'text-amber-400 font-semibold'
                      : 'text-rose-400 font-semibold'
                  }
                >
                  {meanConfidence}% Confidence
                </span>

                {lowConfidenceCount > 0 && (
                  <>
                    <span className="text-slate-600">•</span>
                    <button
                      type="button"
                      onClick={handleJumpToNextAmbiguity}
                      title="Click to jump through low-confidence tokens"
                      data-testid="low-confidence-counter-badge"
                      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30 hover:border-amber-400 transition-all cursor-pointer shadow-sm group"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse group-hover:scale-125 transition-transform" />
                      <span>{lowConfidenceCount} low-confidence tokens</span>
                      <span className="text-[9px] opacity-75 font-normal ml-0.5">(jump)</span>
                    </button>
                  </>
                )}
              </div>

              {/* Action Buttons & Exports */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={handleCopyText}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  {copied ? (
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Copy className="w-3.5 h-3.5 text-slate-300" />
                  )}
                  <span>{copied ? 'Copied' : 'Copy Text'}</span>
                </button>

                <button
                  type="button"
                  onClick={handleExportMarkdown}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  <FileText className="w-3.5 h-3.5 text-slate-300" />
                  <span>Export MD</span>
                </button>

                <button
                  type="button"
                  data-testid="btn-export-searchable-pdf"
                  onClick={handleExportSearchablePdf}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  <span>Export PDF</span>
                </button>

                <button
                  type="button"
                  onClick={handleExportTxt}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  <FileDown className="w-3.5 h-3.5 text-slate-300" />
                  <span>Export TXT</span>
                </button>

                <button
                  type="button"
                  onClick={handleExportJson}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  <Code className="w-3.5 h-3.5 text-slate-300" />
                  <span>Export JSON</span>
                </button>

                {completedBatchDocuments.length > 1 && batchConfig.outputMode === 'single' && (
                  <button
                    type="button"
                    data-testid="btn-export-combined-batch"
                    onClick={handleExportBatchCombined}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/20 border border-white/20 transition-all"
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>Export All ({completedBatchDocuments.length})</span>
                  </button>
                )}
              </div>
            </div>

            {/* Split-Pane Verification Workspace */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0 overflow-hidden">
              {/* Left Column: High-Resolution Scan & Interactive Overlays (7 cols) */}
              <div className="lg:col-span-7 h-full flex flex-col min-h-[500px] overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-2xl shadow-2xl p-3 gap-3">
                {/* View Mode & Darkroom Controls */}
                <div className="flex flex-wrap items-center justify-between gap-2 shrink-0">
                  <div className="flex items-center gap-1 bg-white/[0.06] p-1 rounded-xl border border-white/10 text-xs">
                    <button
                      type="button"
                      data-testid="toggle-view-canvas"
                      onClick={() => setViewMode('canvas')}
                      className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                        viewMode === 'canvas'
                          ? 'bg-blue-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-white'
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
                          ? 'bg-blue-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Split Curtain X-Ray
                    </button>
                  </div>

                  {activePage && (
                    <DarkroomToolbar
                      settings={darkroomSettings}
                      onChange={setDarkroomSettings}
                      className="border-white/10"
                    />
                  )}
                </div>

                {/* Viewer or Curtain */}
                <div className="flex-1 min-h-0 overflow-hidden rounded-2xl relative">
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
                      signatureLineIds={findSignatureCandidates(activePage).map((row) => row.lineId)}
                    />
                  )}
                  {activePage && viewMode === 'curtain' && (
                    <SplitCurtain page={activePage} />
                  )}
                </div>
              </div>

              {/* Right Column: Editable Transcription with Amber Confidence Tokens (5 cols) */}
              <div className="lg:col-span-5 h-full flex flex-col min-h-[500px] overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-2xl shadow-2xl">
                {activePage && (
                  <div className="flex h-full min-h-0 flex-col">
                    <div className="min-h-0 flex-1 overflow-hidden">
                      <InlineEditor
                        page={activePage}
                        documentId={document?.document_id}
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
                    <SignatureInspector
                      page={activePage}
                      reviews={document?.signature_reviews}
                      selectedLineId={selectedLineId}
                      onSelectLine={(lineId) => setSelectedLineId(lineId)}
                      onDecide={setSignatureDecision}
                      className="m-3 mt-0 shrink-0"
                    />
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
