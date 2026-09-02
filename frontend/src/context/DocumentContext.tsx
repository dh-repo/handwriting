'use client';

import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { DocumentOCRResult, PageResult, LineItem, WordToken, SignatureDecision } from '../types/ocr';
import { SAMPLE_PRESETS } from '../lib/sampleDocuments';
import { findSignatureCandidates, upsertSignatureReview } from '../lib/signatureCandidates';

export interface DocumentContextType {
  // Document state
  document: DocumentOCRResult | null;
  activePageIndex: number;
  activePage: PageResult | null;
  setDocument: (doc: DocumentOCRResult | null) => void;
  setActivePageIndex: (index: number) => void;

  // Selection & Hover state (Bidirectional synchronization)
  selectedLineId: string | null;
  selectedWordId: string | null;
  hoveredLineId: string | null;
  hoveredWordId: string | null;
  setSelectedLineId: (lineId: string | null) => void;
  setSelectedWordId: (wordId: string | null, parentLineId?: string) => void;
  setHoveredLineId: (lineId: string | null) => void;
  setHoveredWordId: (wordId: string | null) => void;

  // Viewer Transform State
  scale: number;
  pan: { x: number; y: number };
  rotation: number;
  setScale: React.Dispatch<React.SetStateAction<number>>;
  setPan: React.Dispatch<React.SetStateAction<{ x: number; y: number }>>;
  setRotation: React.Dispatch<React.SetStateAction<number>>;
  zoomIn: (delta?: number) => void;
  zoomOut: (delta?: number) => void;
  resetTransform: () => void;

  // Display Toggles
  showBoundingBoxes: boolean;
  showConfidenceHeatmap: boolean;
  showWordBoxes: boolean;
  confidenceThreshold: number;
  setShowBoundingBoxes: React.Dispatch<React.SetStateAction<boolean>>;
  setShowConfidenceHeatmap: React.Dispatch<React.SetStateAction<boolean>>;
  setShowWordBoxes: React.Dispatch<React.SetStateAction<boolean>>;
  setConfidenceThreshold: (val: number) => void;

  // Processing & Upload State
  isProcessing: boolean;
  uploadProgress: number;
  processingStage: string;
  error: string | null;
  setIsProcessing: (val: boolean) => void;
  setUploadProgress: (val: number) => void;
  setProcessingStage: (stage: string) => void;
  setError: (err: string | null) => void;

  // Text Correction & History
  updateLineText: (lineId: string, newText: string) => void;
  updateWordText: (lineId: string, wordId: string, newText: string) => void;
  revertLine: (lineId: string) => void;
  revertAll: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  loadPreset: (presetId: string) => void;
  setSignatureDecision: (lineId: string, decision: SignatureDecision) => void;
}

const DocumentContext = createContext<DocumentContextType | null>(null);

export const DocumentProvider: React.FC<{ children: React.ReactNode; initialDocument?: DocumentOCRResult | null }> = ({
  children,
  initialDocument = null,
}) => {
  // Document state
  const [document, setDocumentInternal] = useState<DocumentOCRResult | null>(initialDocument);
  const [activePageIndex, setActivePageIndex] = useState<number>(0);

  // Selection / Hover
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [selectedWordId, setSelectedWordIdState] = useState<string | null>(null);
  const [hoveredLineId, setHoveredLineId] = useState<string | null>(null);
  const [hoveredWordId, setHoveredWordId] = useState<string | null>(null);

  // Viewer Transform
  const [scale, setScale] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [rotation, setRotation] = useState<number>(0);

  // Viewer Layer Toggles
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showConfidenceHeatmap, setShowConfidenceHeatmap] = useState<boolean>(true);
  const [showWordBoxes, setShowWordBoxes] = useState<boolean>(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.70);

  // Processing state
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [processingStage, setProcessingStage] = useState<string>('Ready');
  const [error, setError] = useState<string | null>(null);

  // Undo / Redo stacks
  const [undoStack, setUndoStack] = useState<DocumentOCRResult[]>([]);
  const [redoStack, setRedoStack] = useState<DocumentOCRResult[]>([]);

  const activePage = useMemo(() => {
    if (!document || !document.pages || document.pages.length === 0) return null;
    return document.pages[activePageIndex] || document.pages[0] || null;
  }, [document, activePageIndex]);

  const setDocument = useCallback((doc: DocumentOCRResult | null) => {
    setDocumentInternal(doc);
    setActivePageIndex(0);
    setSelectedLineId(null);
    setSelectedWordIdState(null);
    setHoveredLineId(null);
    setHoveredWordId(null);
    setScale(1.0);
    setPan({ x: 0, y: 0 });
    setRotation(0);
    setUndoStack([]);
    setRedoStack([]);
    setError(null);
  }, []);

  const setSelectedWordId = useCallback((wordId: string | null, parentLineId?: string) => {
    setSelectedWordIdState(wordId);
    if (parentLineId) {
      setSelectedLineId(parentLineId);
    }
  }, []);

  const zoomIn = useCallback((delta = 0.2) => {
    setScale((prev) => Math.min(5.0, +(prev + delta).toFixed(2)));
  }, []);

  const zoomOut = useCallback((delta = 0.2) => {
    setScale((prev) => Math.max(0.2, +(prev - delta).toFixed(2)));
  }, []);

  const resetTransform = useCallback(() => {
    setScale(1.0);
    setPan({ x: 0, y: 0 });
    setRotation(0);
  }, []);

  // Max history depth to cap memory growth on large multi-page documents
  const MAX_HISTORY_DEPTH = 50;

  // Push new state with history tracking
  const pushDocumentUpdate = useCallback((newDoc: DocumentOCRResult) => {
    if (document) {
      setUndoStack((prev) => {
        const nextStack = [...prev, structuredClone(document)];
        if (nextStack.length > MAX_HISTORY_DEPTH) {
          return nextStack.slice(nextStack.length - MAX_HISTORY_DEPTH);
        }
        return nextStack;
      });
      setRedoStack([]);
    }
    setDocumentInternal(newDoc);
  }, [document]);

  const updateLineText = useCallback((lineId: string, newText: string) => {
    if (!document) return;

    const newDoc = structuredClone(document);
    for (const page of newDoc.pages) {
      const line = page.lines.find((l: LineItem) => l.line_id === lineId);
      if (line) {
        line.text = newText;
        line.is_edited = newText !== (line.original_text || '');
        // Sync full text
        page.full_text = page.lines.map((l: LineItem) => l.text).join('\n');
        break;
      }
    }
    newDoc.full_text = newDoc.pages.map((p: PageResult) => p.full_text).join('\n\n');
    pushDocumentUpdate(newDoc);
  }, [document, pushDocumentUpdate]);

  const updateWordText = useCallback((lineId: string, wordId: string, newText: string) => {
    if (!document) return;

    const newDoc = structuredClone(document);
    for (const page of newDoc.pages) {
      const line = page.lines.find((l: LineItem) => l.line_id === lineId);
      if (line) {
        const word = line.words.find((w: WordToken) => w.word_id === wordId);
        if (word) {
          word.text = newText;
          word.is_edited = newText !== (word.original_text || '');
          line.text = line.words.map((w: WordToken) => w.text).join(' ');
          line.is_edited = true;
          page.full_text = page.lines.map((l: LineItem) => l.text).join('\n');
          break;
        }
      }
    }
    newDoc.full_text = newDoc.pages.map((p: PageResult) => p.full_text).join('\n\n');
    pushDocumentUpdate(newDoc);
  }, [document, pushDocumentUpdate]);

  const revertLine = useCallback((lineId: string) => {
    if (!document) return;

    const newDoc = structuredClone(document);
    for (const page of newDoc.pages) {
      const line = page.lines.find((l: LineItem) => l.line_id === lineId);
      if (line) {
        line.text = line.original_text || line.text;
        line.is_edited = false;
        line.words.forEach((w: WordToken) => {
          w.text = w.original_text || w.text;
          w.is_edited = false;
        });
        page.full_text = page.lines.map((l: LineItem) => l.text).join('\n');
        break;
      }
    }
    newDoc.full_text = newDoc.pages.map((p: PageResult) => p.full_text).join('\n\n');
    pushDocumentUpdate(newDoc);
  }, [document, pushDocumentUpdate]);

  const revertAll = useCallback(() => {
    if (!document) return;

    const newDoc = structuredClone(document);
    for (const page of newDoc.pages) {
      for (const line of page.lines) {
        line.text = line.original_text || line.text;
        line.is_edited = false;
        line.words.forEach((w: WordToken) => {
          w.text = w.original_text || w.text;
          w.is_edited = false;
        });
      }
      page.full_text = page.lines.map((l: LineItem) => l.text).join('\n');
    }
    newDoc.full_text = newDoc.pages.map((p: PageResult) => p.full_text).join('\n\n');
    pushDocumentUpdate(newDoc);
  }, [document, pushDocumentUpdate]);

  const undo = useCallback(() => {
    if (undoStack.length === 0 || !document) return;
    const previous = undoStack[undoStack.length - 1];
    setUndoStack((prev) => prev.slice(0, prev.length - 1));
    setRedoStack((prev) => [structuredClone(document), ...prev]);
    setDocumentInternal(previous);
  }, [undoStack, document]);

  const redo = useCallback(() => {
    if (redoStack.length === 0 || !document) return;
    const next = redoStack[0];
    setRedoStack((prev) => prev.slice(1));
    setUndoStack((prev) => [...prev, structuredClone(document)]);
    setDocumentInternal(next);
  }, [redoStack, document]);

  const loadPreset = useCallback((presetId: string) => {
    if (SAMPLE_PRESETS[presetId]) {
      setDocument(structuredClone(SAMPLE_PRESETS[presetId]));
    }
  }, [setDocument]);

  const setSignatureDecision = useCallback((lineId: string, decision: SignatureDecision) => {
    if (!document || !activePage) return;
    const candidate = findSignatureCandidates(activePage).find((row) => row.lineId === lineId);
    if (!candidate) return;
    const newDoc = structuredClone(document);
    newDoc.signature_reviews = upsertSignatureReview(newDoc.signature_reviews ?? [], {
      page_number: activePage.page_number,
      line_id: lineId,
      kind: candidate.kind,
      decision,
      decided_at: decision === 'pending' ? undefined : new Date().toISOString(),
    });
    pushDocumentUpdate(newDoc);
  }, [document, activePage, pushDocumentUpdate]);

  const value = useMemo(
    () => ({
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
      scale,
      pan,
      rotation,
      setScale,
      setPan,
      setRotation,
      zoomIn,
      zoomOut,
      resetTransform,
      showBoundingBoxes,
      showConfidenceHeatmap,
      showWordBoxes,
      confidenceThreshold,
      setShowBoundingBoxes,
      setShowConfidenceHeatmap,
      setShowWordBoxes,
      setConfidenceThreshold,
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
      revertLine,
      revertAll,
      undo,
      redo,
      canUndo: undoStack.length > 0,
      canRedo: redoStack.length > 0,
      loadPreset,
      setSignatureDecision,
    }),
    [
      document,
      activePageIndex,
      activePage,
      setDocument,
      selectedLineId,
      selectedWordId,
      hoveredLineId,
      hoveredWordId,
      setSelectedWordId,
      scale,
      pan,
      rotation,
      zoomIn,
      zoomOut,
      resetTransform,
      showBoundingBoxes,
      showConfidenceHeatmap,
      showWordBoxes,
      confidenceThreshold,
      isProcessing,
      uploadProgress,
      processingStage,
      error,
      updateLineText,
      updateWordText,
      revertLine,
      revertAll,
      undo,
      redo,
      undoStack.length,
      redoStack.length,
      loadPreset,
      setSignatureDecision,
    ]
  );

  return <DocumentContext.Provider value={value}>{children}</DocumentContext.Provider>;
};

export function useDocumentContext(): DocumentContextType {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocumentContext must be used within a DocumentProvider');
  }
  return context;
}
