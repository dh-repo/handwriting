'use client';

import React, { useState } from 'react';
import { Copy, Check, FileText, FileSpreadsheet, Code, Download } from 'lucide-react';
import { DocumentOCRResult } from '../types/ocr';
import {
  exportDocumentAsJson,
  exportDocumentAsTxt,
  exportDocumentAsCsv,
  copyTextToClipboard,
  downloadFile,
} from '../lib/exportUtils';

export interface ExportToolbarProps {
  document: DocumentOCRResult;
  className?: string;
}

export const ExportToolbar: React.FC<ExportToolbarProps> = ({ document, className = '' }) => {
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);

  const handleExportTxt = () => {
    const content = exportDocumentAsTxt(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_transcription.txt`;
    downloadFile(content, filename, 'text/plain;charset=utf-8');
  };

  const handleExportJson = () => {
    const content = exportDocumentAsJson(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_ocr_result.json`;
    downloadFile(content, filename, 'application/json');
  };

  const handleExportCsv = () => {
    const content = exportDocumentAsCsv(document);
    const filename = `${document.filename ? document.filename.replace(/\.[^/.]+$/, '') : 'document'}_tokens.csv`;
    downloadFile(content, filename, 'text/csv;charset=utf-8');
  };

  const handleCopyClipboard = async () => {
    const content = exportDocumentAsTxt(document);
    const success = await copyTextToClipboard(content);
    if (success) {
      setCopiedFormat('txt');
      setTimeout(() => setCopiedFormat(null), 2000);
    }
  };

  return (
    <div
      data-testid="export-toolbar-container"
      className={`flex flex-wrap items-center gap-2 p-2.5 rounded-2xl bg-slate-900/70 backdrop-blur-xl border border-slate-800/90 shadow-lg ${className}`}
    >
      <button
        data-testid="btn-copy-clipboard"
        onClick={handleCopyClipboard}
        className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800/90 text-slate-200 hover:text-white hover:bg-slate-700/90 border border-slate-700/80 hover:border-cyan-500/40 shadow-sm transition-all duration-200"
      >
        {copiedFormat === 'txt' ? (
          <Check data-testid="icon-check" className="w-3.5 h-3.5 text-emerald-400 animate-in zoom-in" />
        ) : (
          <Copy data-testid="icon-copy" className="w-3.5 h-3.5 text-slate-400" />
        )}
        <span>{copiedFormat === 'txt' ? 'Copied to Clipboard!' : 'Copy Text'}</span>
      </button>

      <div className="w-px h-5 bg-slate-800 mx-0.5" />

      <button
        data-testid="btn-export-txt"
        onClick={handleExportTxt}
        className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800/90 text-slate-200 hover:text-white hover:bg-slate-700/90 border border-slate-700/80 hover:border-indigo-500/40 shadow-sm transition-all duration-200"
      >
        <FileText className="w-3.5 h-3.5 text-indigo-400" />
        <span>TXT</span>
      </button>

      <button
        data-testid="btn-export-csv"
        onClick={handleExportCsv}
        className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800/90 text-slate-200 hover:text-white hover:bg-slate-700/90 border border-slate-700/80 hover:border-emerald-500/40 shadow-sm transition-all duration-200"
      >
        <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
        <span>CSV (Tokens)</span>
      </button>

      <button
        data-testid="btn-export-json"
        onClick={handleExportJson}
        className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800/90 text-slate-200 hover:text-white hover:bg-slate-700/90 border border-slate-700/80 hover:border-purple-500/40 shadow-sm transition-all duration-200"
      >
        <Code className="w-3.5 h-3.5 text-purple-400" />
        <span>JSON Payload</span>
      </button>
    </div>
  );
};
