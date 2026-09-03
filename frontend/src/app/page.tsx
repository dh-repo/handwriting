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
  Layers,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from 'lucide-react';
import { useDocumentContext } from '../context/DocumentContext';
import { Dropzone } from '../components/Dropzone';
import { DocumentViewer } from '../components/DocumentViewer';
import { InlineEditor } from '../components/InlineEditor';
import { SignatureInspector } from '../components/SignatureInspector';
import { DocumentOCRResult } from '../types/ocr';
import { findSignatureCandidates } from '../lib/signatureCandidates';
import { apiClient } from '../lib/apiClient';
import {
  exportDocumentAsTxt,
  exportDocumentAsJson,
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

  const activeObjectUrlRef = useRef<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isLoupeActive, setIsLoupeActive] = useState(false);

  const cleanupObjectURL = useCallback(() => {
    if (activeObjectUrlRef.current) {
      try {
        URL.revokeObjectURL(activeObjectUrlRef.current);
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

  const handleFileAccepted = async (file: File) => {
    setIsProcessing(true);
    setUploadProgress(15);
    setProcessingStage('Uploading document & preparing preprocessor...');
    setError(null);

    cleanupObjectURL();
    const previewUrl = file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined;
    if (previewUrl) {
      activeObjectUrlRef.current = previewUrl;
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
  };

  const handleReset = () => {
    cleanupObjectURL();
    setDocument(null);
    setError(null);
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

  // Compute metrics
  const totalWords = document
    ? document.pages.reduce(
        (acc, p) => acc + p.lines.reduce((lAcc, l) => lAcc + (l.words?.length || l.text.split(' ').filter(Boolean).length), 0),
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

  return (
    <div className="min-h-screen bg-[#07090E] text-slate-100 flex flex-col font-sans selection:bg-blue-500/30 selection:text-blue-200 antialiased">
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
        {!document ? (
          /* Landing Screen (Apple HIG Minimalism) */
          <div className="flex-1 flex flex-col items-center justify-center py-12 sm:py-20 max-w-3xl mx-auto w-full text-center space-y-8">
            <div className="space-y-3">
              <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-medium text-blue-400 bg-blue-500/10 border border-blue-500/20 backdrop-blur-md">
                <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                <span>Deep Vision-Language Intelligence</span>
              </div>
              <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.15]">
                Transcribe Handwriting with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400">Neural Precision</span>
              </h1>
              <p className="text-sm sm:text-base text-slate-400 max-w-xl mx-auto font-normal leading-relaxed">
                Effortlessly read cursive correspondence, archival notes, signatures, and multi-page documents.
              </p>
            </div>

            {/* Dropzone Component */}
            <Dropzone
              onFileAccepted={handleFileAccepted}
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
        ) : (
          /* Workspace Screen: Document Viewer + Transcription Inspector */
          <div className="flex-1 flex flex-col gap-4 min-h-0">
            {/* Top Document Summary Bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-2xl shadow-xl">
              <div className="flex items-center gap-4 text-xs text-slate-300">
                <span className="font-medium text-white truncate max-w-xs">{document.filename || 'Document'}</span>
                <span className="text-slate-600">•</span>
                <span>{totalLines} {totalLines === 1 ? 'Line' : 'Lines'}</span>
                <span className="text-slate-600">•</span>
                <span>{totalWords} Words</span>
                <span className="text-slate-600">•</span>
                <span className="text-emerald-400 font-medium">{meanConfidence}% Confidence</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyText}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-300" />}
                  <span>{copied ? 'Copied' : 'Copy Text'}</span>
                </button>
                <button
                  onClick={handleExportTxt}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  <FileDown className="w-3.5 h-3.5 text-slate-300" />
                  <span>Export TXT</span>
                </button>
                <button
                  onClick={handleExportJson}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.08] hover:bg-white/[0.14] text-white border border-white/10 transition-all duration-150"
                >
                  <Code className="w-3.5 h-3.5 text-slate-300" />
                  <span>Export JSON</span>
                </button>
              </div>
            </div>

            {/* Split Workspace */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0 overflow-hidden">
              {/* Left Column: Interactive Document Viewer (7 cols) */}
              <div className="lg:col-span-7 h-full flex flex-col min-h-[500px] overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-2xl shadow-2xl">
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
                    signatureLineIds={findSignatureCandidates(activePage).map((row) => row.lineId)}
                  />
                )}
              </div>

              {/* Right Column: Inline Text Editor (5 cols) */}
              <div className="lg:col-span-5 h-full flex flex-col min-h-[500px] overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-2xl shadow-2xl">
                {activePage && (
                  <div className="flex h-full min-h-0 flex-col">
                    <div className="min-h-0 flex-1 overflow-hidden">
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
                    </div>
                    <SignatureInspector
                      page={activePage}
                      reviews={document.signature_reviews}
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
