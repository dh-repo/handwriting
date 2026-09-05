'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Undo2,
  Redo2,
  RotateCcw,
  Check,
  FastForward,
  Layers,
  FileText,
  AlertTriangle,
  Pill,
  Clock,
  Sparkles,
  Loader2,
} from 'lucide-react';
import {
  PageResult,
  LineItem,
  WordToken,
  LowConfidenceWordItem,
  FeedbackSubmissionRequest,
  FeedbackSubmissionResponse,
  FeedbackSyncStatus,
} from '../types/ocr';
import { getConfidenceColor } from '../lib/colorUtils';
import { extractLineCropBase64 } from '../lib/cropUtils';
import { apiClient, ApiClient } from '../lib/apiClient';
import {
  searchMedicalLexicon,
  AutocompleteSuggestion,
  MedicalCategory,
} from '../lib/medicalLexicon';

export interface InlineEditorProps {
  page: PageResult;
  enableMedicalSuggestions?: boolean;
  documentId?: string;
  onPageUpdate?: (updatedPage: PageResult) => void;
  selectedLineId?: string | null;
  selectedWordId?: string | null;
  hoveredLineId?: string | null;
  hoveredWordId?: string | null;
  onSelectLine?: (lineId: string) => void;
  onSelectWord?: (wordId: string, parentLineId?: string) => void;
  onHoverLine?: (lineId: string | null) => void;
  onHoverWord?: (wordId: string | null) => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onRevertAll?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onLineChange?: (lineId: string, newText: string) => void;
  onWordChange?: (lineId: string, wordId: string, newText: string) => void;
  onFeedbackDispatched?: (
    request: FeedbackSubmissionRequest,
    response?: FeedbackSubmissionResponse
  ) => void;
  apiClientInstance?: ApiClient;
  className?: string;
}

export const InlineEditor: React.FC<InlineEditorProps> = ({
  page,
  enableMedicalSuggestions = false,  documentId,
  onPageUpdate,
  selectedLineId,
  selectedWordId,
  hoveredLineId,
  hoveredWordId,
  onSelectLine,
  onSelectWord,
  onHoverLine,
  onHoverWord,
  onUndo,
  onRedo,
  onRevertAll,
  canUndo = false,
  canRedo = false,
  onLineChange,
  onWordChange,
  onFeedbackDispatched,
  apiClientInstance,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'structured' | 'raw' | 'speed_review'>('structured');
  const [confidenceThreshold] = useState(0.70);

  // Local optimistic text map and Flywheel feedback sync status
  const [localLineTexts, setLocalLineTexts] = useState<Record<string, string>>({});
  const [feedbackStatusMap, setFeedbackStatusMap] = useState<Record<string, FeedbackSyncStatus>>({});
  const debounceTimersRef = useRef<Record<string, NodeJS.Timeout>>({});
  const latestEditsRef = useRef<Record<string, string>>({});

  // Cleanup pending debounce timers on unmount
  useEffect(() => {
    return () => {
      Object.values(debounceTimersRef.current).forEach((t) => clearTimeout(t));
    };
  }, []);

  // Synchronize selection scroll into view
  useEffect(() => {
    if (selectedWordId && typeof document !== 'undefined') {
      const el = document.querySelector(`[data-testid="word-chip-${selectedWordId}"]`);
      if (el && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [selectedWordId]);

  // Active word editing state in structured mode
  const [editingWordId, setEditingWordId] = useState<string | null>(null);
  const [editingWordText, setEditingWordText] = useState<string>('');
  const [editingLineId, setEditingLineId] = useState<string | null>(null);

  // Autocomplete state
  const [suggestions, setSuggestions] = useState<AutocompleteSuggestion[]>([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState<number>(0);
  const [showPopover, setShowPopover] = useState<boolean>(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const wordInputRef = useRef<HTMLInputElement>(null);

  // Speed Review state
  const [speedQueue, setSpeedQueue] = useState<LowConfidenceWordItem[]>([]);
  const [speedIndex, setSpeedIndex] = useState(0);
  const [speedInput, setSpeedInput] = useState('');
  const [speedInputTouched, setSpeedInputTouched] = useState(false);
  const [speedSuggestions, setSpeedSuggestions] = useState<AutocompleteSuggestion[]>([]);
  const speedInputRef = useRef<HTMLInputElement>(null);

  // Sync low-confidence words and unverified proper nouns for Speed Review
  useEffect(() => {
    const queue: LowConfidenceWordItem[] = [];
    page.lines.forEach((line: LineItem) => {
      line.words?.forEach((word: WordToken) => {
        const isProper = Boolean(
          word.is_proper_noun ||
            /^[A-Z][a-zA-Z'\-]*\.?$/.test(word.text || '') ||
            /^[A-Z]\.?$/.test(word.text || '')
        );
        // Include words below general threshold OR proper nouns below high-confidence safety threshold (0.90)
        if (!word.reviewed && (word.confidence == null || (word.confidence ?? 0) < confidenceThreshold || (isProper && (word.confidence ?? 0) < 0.90))) {
          queue.push({
            page_index: page.page_number - 1,
            line_id: line.line_id,
            word_id: word.word_id,
            text: word.text,
            original_text: word.original_text || word.text,
            confidence: word.confidence,
            bbox: word.bbox,
            page_image_url: page.image_url,
            alternatives: word.alternatives,
            is_proper_noun: isProper,
          });
        }
      });
    });


    setSpeedQueue(queue);
    if (speedIndex >= queue.length) setSpeedIndex(0);
  }, [page, confidenceThreshold, speedIndex]);

  // Update speed review active word & auto-suggestions
  useEffect(() => {
    if (activeTab === 'speed_review' && speedQueue[speedIndex]) {
      const activeWord = speedQueue[speedIndex];
      setSpeedInput(activeWord.text);
      setSpeedInputTouched(false);
      onSelectLine?.(activeWord.line_id);
      onSelectWord?.(activeWord.word_id, activeWord.line_id);

      // Search medical lexicon for top suggestions
      const results: AutocompleteSuggestion[] = enableMedicalSuggestions ? searchMedicalLexicon({
        query: activeWord.text,
        maxResults: 5,
        enableFuzzy: true,
      }) : [];
      setSpeedSuggestions(results);

      setTimeout(() => speedInputRef.current?.focus(), 50);
    }
  }, [activeTab, speedIndex, speedQueue, onSelectLine, onSelectWord]);

  // Update autocomplete suggestions when editing a word in structured mode
  useEffect(() => {
    if (editingWordId && editingWordText.trim().length > 0) {
      const results: AutocompleteSuggestion[] = enableMedicalSuggestions ? searchMedicalLexicon({
        query: editingWordText,
        maxResults: 6,
        enableFuzzy: true,
      }) : [];
      setSuggestions(results);
      setSelectedSuggestionIndex(0);
      setShowPopover(results.length > 0);
    } else {
      setSuggestions([]);
      setShowPopover(false);
    }
  }, [editingWordId, editingWordText]);

  const dispatchFeedbackForLine = useCallback(
    async (lineId: string, textOverride?: string) => {
      if (debounceTimersRef.current[lineId]) {
        clearTimeout(debounceTimersRef.current[lineId]);
        delete debounceTimersRef.current[lineId];
      }

      const line = page.lines.find((l) => l.line_id === lineId);
      if (!line || documentId === "demo") return;

      const currentText =
        textOverride !== undefined
          ? textOverride
          : latestEditsRef.current[lineId] ?? localLineTexts[lineId] ?? line.text;

      const originalText = line.original_text || line.text || '';
      if (currentText === originalText && !line.is_edited) {
        setFeedbackStatusMap((prev) => ({ ...prev, [lineId]: 'idle' }));
        return;
      }

      setFeedbackStatusMap((prev) => ({ ...prev, [lineId]: 'syncing' }));

      let cropBase64: string | undefined;
      try {
        cropBase64 = await extractLineCropBase64(page.image_url, line.bbox);
      } catch {
        cropBase64 = undefined;
      }

      const payload: FeedbackSubmissionRequest = {
        document_id: documentId || (page as any).document_id || `doc_p${page.page_number}`,
        page_number: page.page_number,
        line_id: lineId,
        original_prediction: originalText,
        operator_correction: currentText,
        original_text: originalText,
        corrected_text: currentText,
        confidence: line.confidence,
        bbox: line.bbox,
        line_crop_base64: cropBase64,
        timestamp: new Date().toISOString(),
      };

      try {
        const client = apiClientInstance || apiClient;
        const response = await client.submitFeedback(payload);
        setFeedbackStatusMap((prev) => ({ ...prev, [lineId]: 'synced' }));
        onFeedbackDispatched?.(payload, response);
      } catch (err) {
        console.error('[InlineEditor] Line feedback dispatch failed:', err);
        setFeedbackStatusMap((prev) => ({ ...prev, [lineId]: 'error' }));
      }
    },
    [page, documentId, apiClientInstance, onFeedbackDispatched, localLineTexts]
  );

  const dispatchFeedbackForWord = useCallback(
    async (args: {
      line_id: string;
      word_id?: string;
      original_text: string;
      corrected_text: string;
      confidence?: number | null;
      bbox?: [number, number, number, number];
    }) => {
      const { line_id, word_id, original_text, corrected_text, confidence, bbox } = args;
      if (original_text === corrected_text || documentId === "demo") return;

      setFeedbackStatusMap((prev) => ({ ...prev, [line_id]: 'syncing' }));

      let cropBase64: string | undefined;
      try {
        const line = page.lines.find((l) => l.line_id === line_id);
        cropBase64 = await extractLineCropBase64(page.image_url, bbox || line?.bbox);
      } catch {
        cropBase64 = undefined;
      }

      const payload: FeedbackSubmissionRequest = {
        document_id: documentId || (page as any).document_id || `doc_p${page.page_number}`,
        page_number: page.page_number,
        line_id,
        word_id,
        original_prediction: original_text,
        operator_correction: corrected_text,
        original_text,
        corrected_text,
        confidence,
        bbox,
        line_crop_base64: cropBase64,
        timestamp: new Date().toISOString(),
      };

      try {
        const client = apiClientInstance || apiClient;
        const response = await client.submitFeedback(payload);
        setFeedbackStatusMap((prev) => ({ ...prev, [line_id]: 'synced' }));
        onFeedbackDispatched?.(payload, response);
      } catch (err) {
        console.error('[InlineEditor] Word feedback dispatch failed:', err);
        setFeedbackStatusMap((prev) => ({ ...prev, [line_id]: 'error' }));
      }
    },
    [page, documentId, apiClientInstance, onFeedbackDispatched]
  );

  const handleLineTextChange = (lineId: string, newText: string) => {
    if (onLineChange) {
      onLineChange(lineId, newText);
      return;
    }
    if (onPageUpdate) {
      const updatedLines = page.lines.map((line: LineItem) => {
        if (line.line_id !== lineId) return line;
        return {
          ...line,
          text: newText,
          is_edited: newText !== (line.original_text || line.text),
        };
      });
      onPageUpdate({
        ...page,
        lines: updatedLines,
        full_text: updatedLines.map((l: LineItem) => l.text).join('\n'),
      });
    }
  };

  const handleLineInputChange = (lineId: string, newText: string) => {
    setLocalLineTexts((prev) => ({ ...prev, [lineId]: newText }));
    latestEditsRef.current[lineId] = newText;
    handleLineTextChange(lineId, newText);

    if (debounceTimersRef.current[lineId]) {
      clearTimeout(debounceTimersRef.current[lineId]);
    }

    setFeedbackStatusMap((prev) => ({ ...prev, [lineId]: 'debouncing' }));

    debounceTimersRef.current[lineId] = setTimeout(() => {
      dispatchFeedbackForLine(lineId, newText);
    }, 500);
  };

  const handleWordTextChange = useCallback(
    (lineId: string, wordId: string, newText: string) => {
      if (onWordChange) {
        onWordChange(lineId, wordId, newText);
        return;
      }
      if (onPageUpdate) {
        const updatedLines = page.lines.map((line: LineItem) => {
          if (line.line_id !== lineId) return line;
          const updatedWords = (line.words || []).map((w: WordToken) => {
            if (w.word_id !== wordId) return w;
            return {
              ...w,
              text: newText,
              is_edited: newText !== (w.original_text || w.text),
            };
          });
          const reconstructed = updatedWords.map((w: WordToken) => w.text).join(' ');
          return {
            ...line,
            words: updatedWords,
            text: reconstructed,
            is_edited: true,
          };
        });
        onPageUpdate({
          ...page,
          lines: updatedLines,
          full_text: updatedLines.map((l: LineItem) => l.text).join('\n'),
        });
      }
    },
    [onWordChange, onPageUpdate, page]
  );

  const handleApplySuggestion = (suggestion: AutocompleteSuggestion) => {
    if (editingLineId && editingWordId) {
      const line = page.lines.find((l) => l.line_id === editingLineId);
      const word = line?.words.find((w) => w.word_id === editingWordId);
      const origText = word?.original_text || word?.text || editingWordText;

      handleWordTextChange(editingLineId, editingWordId, suggestion.entry.term);
      setEditingWordText(suggestion.entry.term);
      setShowPopover(false);
      setEditingWordId(null);

      dispatchFeedbackForWord({
        line_id: editingLineId,
        word_id: editingWordId,
        original_text: origText,
        corrected_text: suggestion.entry.term,
        confidence: word?.confidence,
        bbox: word?.bbox,
      });
    }
  };

  const handleSpeedSubmit = (customText?: string) => {
    if (!speedQueue[speedIndex]) return;
    const current = speedQueue[speedIndex];
    const textToApply = customText !== undefined ? customText : speedInput;
    handleWordTextChange(current.line_id, current.word_id, textToApply);

    dispatchFeedbackForWord({
      line_id: current.line_id,
      word_id: current.word_id,
      original_text: current.original_text || current.text,
      corrected_text: textToApply,
      confidence: current.confidence,
      bbox: current.bbox,
    });

    setSpeedInputTouched(false);
    if (speedIndex < speedQueue.length - 1) {
      setSpeedIndex((prev) => prev + 1);
    } else {
      setActiveTab('structured');
    }
  };

  const handleSpeedSkip = () => {
    setSpeedInputTouched(false);
    if (speedIndex < speedQueue.length - 1) {
      setSpeedIndex((prev) => prev + 1);
    } else {
      setActiveTab('structured');
    }
  };

  const handleSpeedPrev = () => {
    setSpeedInputTouched(false);
    if (speedIndex > 0) {
      setSpeedIndex((prev) => prev - 1);
    }
  };

  const handleRevertAll = () => {
    Object.values(debounceTimersRef.current).forEach((t) => clearTimeout(t));
    debounceTimersRef.current = {};
    setFeedbackStatusMap({});
    setLocalLineTexts({});
    latestEditsRef.current = {};
    onRevertAll?.();
  };

  const renderFeedbackStatusBadge = (lineId: string) => {
    const status = feedbackStatusMap[lineId];
    if (!status || status === 'idle') return null;

    switch (status) {
      case 'debouncing':
        return (
          <span
            data-testid={`feedback-status-${lineId}`}
            className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 animate-pulse"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-ping mr-0.5" />
            Staged...
          </span>
        );
      case 'syncing':
        return (
          <span
            data-testid={`feedback-status-${lineId}`}
            className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20"
          >
            <Loader2 className="w-3 h-3 animate-spin text-indigo-500" />
            Syncing...
          </span>
        );
      case 'synced':
        return (
          <span
            data-testid={`feedback-status-${lineId}`}
            className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
          >
            <Check className="w-3 h-3 text-emerald-500" />
            Feedback submitted
          </span>
        );
      case 'error':
        return (
          <span
            data-testid={`feedback-status-${lineId}`}
            className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20"
          >
            <AlertTriangle className="w-3 h-3 text-rose-500" />
            <span>Sync Failed</span>
            <button
              type="button"
              data-testid={`retry-feedback-${lineId}`}
              onClick={(e) => {
                e.stopPropagation();
                dispatchFeedbackForLine(lineId);
              }}
              className="ml-1 underline font-bold hover:text-rose-700 dark:hover:text-rose-300"
            >
              Retry
            </button>
          </span>
        );
      default:
        return null;
    }
  };

  const renderCategoryBadge = (category: MedicalCategory) => {
    switch (category) {
      case 'medication':
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-indigo-500/10 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30">
            Medication
          </span>
        );
      case 'dosage':
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
            Dosage
          </span>
        );
      case 'sig_frequency':
      case 'sig_timing':
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 border border-amber-500/30">
            Sig Code
          </span>
        );
      case 'sig_route':
      case 'dosage_form':
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-sky-500/10 dark:bg-sky-950/60 text-sky-600 dark:text-sky-400 border border-sky-500/30">
            Route / Form
          </span>
        );
      case 'instruction':
      default:
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-500/10 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-500/30">
            Instruction
          </span>
        );
    }
  };

  const totalWords = page.lines.reduce((acc, l) => acc + (l.words?.length || 0), 0);
  const lowConfCount = speedQueue.length;
  const editedCount = page.lines.filter((l) => l.is_edited).length;

  return (
    <div
      data-testid="inline-editor-container"
      className={`flex flex-col h-full bg-[#09090b] border border-zinc-800 rounded-3xl shadow-sm overflow-hidden ${className}`}
    >
      {/* Editor Header Toolbar */}
      <div className="flex items-center justify-between p-2.5 border-b border-zinc-800/80 bg-zinc-950/60">
        <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-xl border border-zinc-800">
          <button
            type="button"
            data-testid="tab-structured"
            onClick={() => {
              setActiveTab('structured');
              setEditingWordId(null);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'structured'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span>Lines ({page.lines.length})</span>
          </button>
          <button
            type="button"
            data-testid="tab-raw"
            onClick={() => {
              setActiveTab('raw');
              setEditingWordId(null);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'raw'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <FileText className="w-3.5 h-3.5 text-zinc-400" />
            <span>Plain Text</span>
          </button>
          <button
            type="button"
            data-testid="tab-speed-review"
            onClick={() => {
              setActiveTab('speed_review');
              setEditingWordId(null);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'speed_review'
                ? 'bg-amber-500 text-zinc-950 font-bold shadow-sm'
                : lowConfCount > 0
                ? 'text-amber-400 hover:bg-amber-500/10'
                : 'text-zinc-600 opacity-60'
            }`}
          >
            <FastForward className="w-3.5 h-3.5" />
            <span>Speed Review ({lowConfCount})</span>
          </button>
        </div>

        {/* Undo / Redo / Revert controls */}
        <div className="flex items-center gap-1">
          {onUndo && (
            <button
              type="button"
              data-testid="btn-undo"
              onClick={onUndo}
              disabled={!canUndo}
              title="Undo (Ctrl+Z)"
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60 disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <Undo2 className="w-4 h-4" />
            </button>
          )}
          {onRedo && (
            <button
              type="button"
              data-testid="btn-redo"
              onClick={onRedo}
              disabled={!canRedo}
              title="Redo (Ctrl+Y)"
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60 disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <Redo2 className="w-4 h-4" />
            </button>
          )}
          {onRevertAll && (
            <button
              type="button"
              data-testid="btn-revert-all"
              onClick={handleRevertAll}
              disabled={editedCount === 0}
              title="Revert all edits"
              className="p-1.5 rounded-lg text-zinc-400 hover:text-rose-400 hover:bg-rose-950/40 disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <span data-testid="revert-all-btn" className="inline-flex items-center justify-center pointer-events-none">
                <RotateCcw className="w-4 h-4" />
              </span>
            </button>
          )}
        </div>
      </div>

      {/* Metrics Summary Strip */}
      <div
        data-testid="editor-metrics-strip"
        className="flex items-center justify-between px-3.5 py-1.5 bg-zinc-950/40 border-b border-zinc-800/80 text-[11px] font-mono text-zinc-400"
      >
        <div className="flex items-center gap-3">
          <span>
            Model score:{' '}
            <strong
              className={
                (page.mean_confidence ?? -1) >= 0.90
                  ? 'text-emerald-400'
                  : (page.mean_confidence ?? -1) >= 0.70
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }
            >
              {(page.mean_confidence == null ? "Unknown" : (page.mean_confidence * 100).toFixed(1))}%
            </strong>
          </span>
          <span className="text-zinc-600">•</span>
          <span>
            Words: <strong className="text-zinc-200">{totalWords}</strong>
          </span>
          <span className="text-zinc-600">•</span>
          <span>
            Lines: <strong className="text-zinc-200">{page.lines.length}</strong>
          </span>
        </div>
        {editedCount > 0 && (
          <span data-testid="edited-count-badge" className="text-indigo-400 font-medium font-sans text-xs">
            {editedCount} line(s) modified
          </span>
        )}
      </div>

      {/* Tab Content Container */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-2.5 relative">
        {/* Structured Line-by-Line Mode */}
        {activeTab === 'structured' && (
          <div data-testid="structured-lines-list" className="space-y-2.5">
            {page.lines.map((line: LineItem, idx: number) => {
              const isSelected = selectedLineId === line.line_id;
              const isHovered = hoveredLineId === line.line_id;
              const colorStyle = getConfidenceColor(line.confidence);

              return (
                <div
                  key={line.line_id}
                  data-testid={`line-row-${line.line_id}`}
                  onClick={() => onSelectLine?.(line.line_id)}
                  onMouseEnter={() => onHoverLine?.(line.line_id)}
                  onMouseLeave={() => onHoverLine?.(null)}
                  className={`p-3 rounded-xl border transition-all duration-150 relative ${
                    isSelected
                      ? 'border-indigo-500/70 bg-indigo-950/20 ring-1 ring-indigo-500/30'
                      : isHovered
                      ? 'border-zinc-700 bg-zinc-900/50'
                      : 'border-zinc-800/80 hover:border-zinc-700/80 bg-zinc-900/30'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-semibold">
                        L{idx + 1}
                      </span>
                      <span
                        className={`text-[11px] font-mono font-medium px-1.5 py-0.5 rounded border ${colorStyle.badgeBg} ${colorStyle.badgeBorder} ${colorStyle.tailwindText}`}
                      >
                        {(line.confidence == null ? "Unknown" : (line.confidence * 100).toFixed(0))}%
                      </span>
                      {line.is_edited && (
                        <span
                          data-testid={`edited-badge-${line.line_id}`}
                          className="text-[9px] uppercase font-bold text-indigo-400 bg-indigo-950/60 px-1.5 py-0.5 rounded border border-indigo-800/60"
                        >
                          Edited
                        </span>
                      )}
                      {renderFeedbackStatusBadge(line.line_id)}
                    </div>
                  </div>

                  <input
                    type="text"
                    data-testid={`line-input-${line.line_id}`}
                    value={localLineTexts[line.line_id] ?? line.text}
                    onChange={(e) => handleLineInputChange(line.line_id, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        if (debounceTimersRef.current[line.line_id]) {
                          clearTimeout(debounceTimersRef.current[line.line_id]);
                          delete debounceTimersRef.current[line.line_id];
                        }
                        dispatchFeedbackForLine(line.line_id, e.currentTarget.value);
                      }
                    }}
                    onBlur={(e) => {
                      if (debounceTimersRef.current[line.line_id]) {
                        clearTimeout(debounceTimersRef.current[line.line_id]);
                        delete debounceTimersRef.current[line.line_id];
                        dispatchFeedbackForLine(line.line_id, e.currentTarget.value);
                      }
                    }}
                    className="w-full text-sm font-medium px-3 py-1.5 rounded-lg bg-zinc-950/60 border border-zinc-800 text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                  />

                  {/* Word-level breakdown chips with diff-style underlines */}
                  {line.words && line.words.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1 mt-2 pt-1.5 border-t border-zinc-800/60">
                      {line.words.map((word: WordToken) => {
                        const isWordSelected = selectedWordId === word.word_id;
                        const isWordHovered = hoveredWordId === word.word_id;
                        const isLowConfidence = (word.confidence ?? 0) < 0.85;
                        const isEditingThisWord = editingWordId === word.word_id;
                        const isProperNoun = Boolean(
                          word.is_proper_noun ||
                            /^[A-Z][a-zA-Z'\-]*\.?$/.test(word.text || '') ||
                            /^[A-Z]\.?$/.test(word.text || '')
                        );
                        const isUncertainProperNoun = isProperNoun && (word.confidence ?? 0) < 0.90;

                        return (
                          <div key={word.word_id} className="relative inline-block">
                            {isEditingThisWord ? (
                              <input
                                ref={wordInputRef}
                                type="text"
                                data-testid={`word-edit-input-${word.word_id}`}
                                value={editingWordText}
                                onChange={(e) => setEditingWordText(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'ArrowDown') {
                                    e.preventDefault();
                                    if (suggestions.length > 0) {
                                      setSelectedSuggestionIndex((prev) => (prev + 1) % suggestions.length);
                                    }
                                  } else if (e.key === 'ArrowUp') {
                                    e.preventDefault();
                                    if (suggestions.length > 0) {
                                      setSelectedSuggestionIndex((prev) =>
                                        prev === 0 ? suggestions.length - 1 : prev - 1
                                      );
                                    }
                                  } else if (e.key === 'Enter' || e.key === 'Tab') {
                                    e.preventDefault();
                                    if (showPopover && suggestions[selectedSuggestionIndex]) {
                                      handleApplySuggestion(suggestions[selectedSuggestionIndex]);
                                    } else {
                                      const origText = word.original_text || word.text;
                                      handleWordTextChange(line.line_id, word.word_id, editingWordText);
                                      setEditingWordId(null);
                                      setShowPopover(false);
                                      if (editingWordText !== origText) {
                                        dispatchFeedbackForWord({
                                          line_id: line.line_id,
                                          word_id: word.word_id,
                                          original_text: origText,
                                          corrected_text: editingWordText,
                                          confidence: word.confidence,
                                          bbox: word.bbox,
                                        });
                                      }
                                    }
                                  } else if (e.key === 'Escape') {
                                    setShowPopover(false);
                                    setEditingWordId(null);
                                  }
                                }}
                                onBlur={() => {
                                  setTimeout(() => {
                                    const origText = word.original_text || word.text;
                                    handleWordTextChange(line.line_id, word.word_id, editingWordText);
                                    setEditingWordId(null);
                                    setShowPopover(false);
                                    if (editingWordText !== origText) {
                                      dispatchFeedbackForWord({
                                        line_id: line.line_id,
                                        word_id: word.word_id,
                                        original_text: origText,
                                        corrected_text: editingWordText,
                                        confidence: word.confidence,
                                        bbox: word.bbox,
                                      });
                                    }
                                  }, 200);
                                }}
                                autoFocus
                                className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-indigo-600 text-white border-2 border-indigo-400 outline-none w-28"
                              />
                            ) : (
                              <button
                                type="button"
                                data-testid={`word-chip-${word.word_id}`}
                                data-confidence-low={isLowConfidence ? 'true' : 'false'}
                                data-proper-noun={isProperNoun ? 'true' : 'false'}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onSelectWord?.(word.word_id, line.line_id);
                                  setEditingWordId(word.word_id);
                                  setEditingLineId(line.line_id);
                                  setEditingWordText(word.text);
                                }}
                                onMouseEnter={() => onHoverWord?.(word.word_id)}
                                onMouseLeave={() => onHoverWord?.(null)}
                                className={`cursor-pointer px-1.5 py-0.5 rounded text-xs font-mono transition-all text-left ${
                                  isWordSelected
                                    ? 'bg-indigo-600 text-white font-bold ring-2 ring-indigo-400 shadow-md'
                                    : isWordHovered
                                    ? 'bg-zinc-800 text-zinc-100 ring-1 ring-zinc-600'
                                    : isUncertainProperNoun
                                    ? 'text-amber-200 bg-amber-500/10 border-b-2 border-amber-400 border-dashed rounded-t font-semibold'
                                    : isLowConfidence
                                    ? 'text-amber-200 bg-amber-500/10 border-b-2 border-amber-400 border-dashed rounded-t'
                                    : isProperNoun
                                    ? 'text-indigo-200 bg-indigo-500/10 border-b-2 border-indigo-400 border-dotted rounded-t'
                                    : 'text-zinc-300 hover:text-white hover:bg-zinc-800/60 border border-transparent'
                                }`}
                                title={
                                  isUncertainProperNoun
                                    ? `Unverified proper noun / name: ${(word.confidence == null ? "Unknown" : (word.confidence * 100).toFixed(0))}% confidence • Click to verify exact spelling`
                                    : isLowConfidence
                                    ? `Low confidence token: ${(word.confidence == null ? "Unknown" : (word.confidence * 100).toFixed(0))}% • Click to correct`
                                    : `${(word.confidence == null ? "Unknown" : (word.confidence * 100).toFixed(0))}% confidence`
                                }
                              >
                                <span className="inline-flex items-center gap-1">
                                  {isUncertainProperNoun && !isWordSelected && (
                                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping flex-shrink-0" title="Verify name/proper noun" />
                                  )}
                                  {isLowConfidence && !isUncertainProperNoun && !isWordSelected && (
                                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse flex-shrink-0" />
                                  )}
                                  <span>{word.text}</span>
                                  {isProperNoun && (
                                    <span
                                      data-testid={`proper-noun-tag-${word.word_id}`}
                                      className="text-[9px] uppercase px-1 py-0.2 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/60 font-sans font-semibold tracking-wider ml-0.5"
                                    >
                                      name
                                    </span>
                                  )}
                                </span>
                              </button>
                            )}

                            {/* Floating Autocomplete Popover */}
                            {isEditingThisWord && showPopover && suggestions.length > 0 && (
                              <div
                                ref={popoverRef}
                                data-testid="medical-autocomplete-popover"
                                className="absolute top-full left-0 mt-1 z-50 w-72 max-h-64 overflow-y-auto bg-slate-950/95 backdrop-blur border border-indigo-500/50 rounded-xl shadow-2xl p-1 space-y-1 text-xs animate-in fade-in zoom-in-95"
                              >
                                <div className="px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center justify-between border-b border-slate-800 pb-1">
                                  <span className="flex items-center gap-1">
                                    <Sparkles className="w-3 h-3 text-indigo-400" /> Medical Suggestions
                                  </span>
                                  <span className="font-mono text-slate-500">↑↓ Nav • ↵ Select</span>
                                </div>
                                {suggestions.map((s, sIdx) => {
                                  const isHighlighted = selectedSuggestionIndex === sIdx;
                                  return (
                                    <div
                                      key={s.entry.term}
                                      data-testid={`autocomplete-item-${sIdx}`}
                                      onMouseDown={(e) => {
                                        e.preventDefault();
                                        handleApplySuggestion(s);
                                      }}
                                      className={`p-2 rounded-lg cursor-pointer transition-colors ${
                                        isHighlighted
                                          ? 'bg-indigo-600 text-white'
                                          : 'hover:bg-slate-800/80 text-slate-200'
                                      }`}
                                    >
                                      <div className="flex items-center justify-between mb-0.5">
                                        <span className="font-bold text-sm">{s.entry.term}</span>
                                        {renderCategoryBadge(s.entry.category)}
                                      </div>
                                      <p className="text-[11px] text-slate-300 line-clamp-1">
                                        {s.entry.description}
                                      </p>
                                      {s.entry.isLasa && (
                                        <div className="flex items-center gap-1 text-[10px] text-amber-400 mt-1">
                                          <AlertTriangle className="w-3 h-3 text-amber-400 flex-shrink-0" />
                                          <span className="truncate">
                                            LASA Caution: {s.entry.lasaWarning || `Confusable with ${s.entry.lasaConfusionWith}`}
                                          </span>
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Plain Text Mode */}
        {activeTab === 'raw' && (
          <textarea
            data-testid="raw-textarea"
            value={page.full_text}
            onChange={(e) => {
              if (onPageUpdate) {
                const lines = e.target.value.split('\n');
                const updatedLines = page.lines.map((l: LineItem, i: number) => ({
                  ...l,
                  text: lines[i] !== undefined ? lines[i] : l.text,
                  is_edited: true,
                }));
                onPageUpdate({ ...page, full_text: e.target.value, lines: updatedLines });
              }
            }}
            rows={16}
            className="w-full h-full p-4 font-mono text-sm rounded-xl bg-zinc-950/60 border border-zinc-800 text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none leading-relaxed"
          />
        )}

        {/* Speed Review Mode */}
        {activeTab === 'speed_review' && (
          <div
            data-testid="speed-review-panel"
            className="flex flex-col items-center justify-center p-6 space-y-6 focus:outline-none"
            onKeyDown={(e) => {
              if (
                e.target !== speedInputRef.current &&
                e.key >= '1' &&
                e.key <= '5'
              ) {
                const num = parseInt(e.key, 10);
                if (speedSuggestions[num - 1]) {
                  e.preventDefault();
                  handleSpeedSubmit(speedSuggestions[num - 1].entry.term);
                }
              }
            }}
          >
            {speedQueue.length === 0 ? (
              <div className="text-center py-12 space-y-3">
                <div className="w-12 h-12 rounded-full bg-emerald-950/50 flex items-center justify-center text-emerald-400 mx-auto border border-emerald-800/50">
                  <Check className="w-6 h-6" />
                </div>
                <h3 className="font-semibold text-lg text-zinc-100">All Words Verified!</h3>
                <p className="text-xs text-zinc-400">
                  No uncertain words found below {(confidenceThreshold * 100).toFixed(0)}% confidence.
                </p>
                <button
                  type="button"
                  data-testid="btn-return-lines"
                  onClick={() => setActiveTab('structured')}
                  className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-500 transition-colors shadow-sm"
                >
                  Return to Line Editor
                </button>
              </div>
            ) : (
              <div className="w-full max-w-lg bg-zinc-900/80 border border-zinc-800 rounded-2xl p-6 shadow-xl space-y-5">
                <div className="flex items-center justify-between text-xs text-zinc-400">
                  <span>
                    Reviewing Uncertain Word <strong>{speedIndex + 1}</strong> of <strong>{speedQueue.length}</strong>
                  </span>
                  <span className="px-2 py-0.5 rounded-full font-mono font-bold bg-rose-950/60 text-rose-300 border border-rose-800/60 text-[11px]">
                    {(speedQueue[speedIndex].confidence == null ? "Unknown" : (speedQueue[speedIndex].confidence! * 100).toFixed(1))}% Conf
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-zinc-950/80 border border-zinc-800 text-center space-y-1">
                  <span className="text-xs text-zinc-500 block font-mono">Model OCR Prediction:</span>
                  <span className="font-mono text-2xl font-bold text-zinc-100">
                    &ldquo;{speedQueue[speedIndex].original_text}&rdquo;
                  </span>
                </div>

                {/* Direct Medical Quick-Select Suggestions (1-5) */}
                {speedSuggestions.length > 0 && (
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-zinc-400 flex items-center gap-1">
                      <Pill className="w-3.5 h-3.5 text-indigo-400" /> Quick-Pick Top Suggestions (Keys 1-5):
                    </span>
                    <div className="grid grid-cols-1 gap-1.5">
                      {speedSuggestions.map((s, idx) => (
                        <button
                          key={s.entry.term}
                          type="button"
                          data-testid={`speed-suggestion-${idx + 1}`}
                          onClick={() => handleSpeedSubmit(s.entry.term)}
                          className="flex items-center justify-between p-2 rounded-xl bg-zinc-950 border border-zinc-800 hover:border-indigo-500 hover:bg-zinc-800/60 text-left transition-all group"
                        >
                          <div className="flex items-center gap-2">
                            <span className="w-5 h-5 rounded-md bg-zinc-800 text-indigo-300 font-mono font-bold text-xs flex items-center justify-center border border-zinc-700">
                              {idx + 1}
                            </span>
                            <span className="font-bold text-sm text-zinc-100">
                              {s.entry.term}
                            </span>
                            {renderCategoryBadge(s.entry.category)}
                          </div>
                          <span className="text-xs text-zinc-400 group-hover:text-zinc-200 max-w-[180px] truncate">
                            {s.entry.description}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <label htmlFor="speed-review-input" className="block text-xs font-semibold text-zinc-300 mb-1.5">
                    Your Correction:
                  </label>
                  <input
                    id="speed-review-input"
                    ref={speedInputRef}
                    data-testid="speed-review-input"
                    type="text"
                    value={speedInput}
                    onChange={(e) => {
                      setSpeedInput(e.target.value);
                      setSpeedInputTouched(true);
                    }}
                    onKeyDown={(e) => {
                      if (e.key >= '1' && e.key <= '5') {
                        const isUntouchedOrEmpty = !speedInputTouched || speedInput.trim() === '';
                        if (e.altKey || isUntouchedOrEmpty) {
                          const num = parseInt(e.key, 10);
                          if (speedSuggestions[num - 1]) {
                            e.preventDefault();
                            handleSpeedSubmit(speedSuggestions[num - 1].entry.term);
                            return;
                          }
                        }
                      }
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleSpeedSubmit();
                      } else if (e.key === 'Tab') {
                        e.preventDefault();
                        if (e.shiftKey) {
                          handleSpeedPrev();
                        } else {
                          handleSpeedSkip();
                        }
                      } else if (e.key === 'Escape') {
                        setActiveTab('structured');
                      }
                    }}
                    className="w-full text-base font-semibold px-4 py-2.5 rounded-xl bg-zinc-950 border border-indigo-500 text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
                  />
                </div>

                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      data-testid="btn-speed-prev"
                      onClick={handleSpeedPrev}
                      disabled={speedIndex === 0}
                      className="px-3 py-2 rounded-xl text-xs font-medium text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors disabled:opacity-30 disabled:pointer-events-none"
                    >
                      Prev
                    </button>
                    <button
                      type="button"
                      data-testid="btn-speed-skip"
                      onClick={handleSpeedSkip}
                      className="px-4 py-2 rounded-xl text-xs font-medium text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors"
                    >
                      Skip (Tab)
                    </button>
                  </div>
                  <button
                    type="button"
                    data-testid="btn-speed-accept"
                    onClick={() => handleSpeedSubmit()}
                    className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition-colors"
                  >
                    Accept & Next (Enter)
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
