import { DocumentOCRResult } from './ocr';

export type OutputMode = 'single' | 'discrete';
export type ModelBias = 'general' | 'archival' | 'tabular';
export type BatchExportFormat = 'markdown' | 'text' | 'json' | 'pdf';

export interface BatchConfiguration {
  outputMode: OutputMode;
  modelBias: ModelBias;
  format: BatchExportFormat;
}

export type BatchItemStatus = 'staged' | 'queued' | 'processing' | 'completed' | 'error';

export interface BatchFileItem {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  previewUrl?: string;
  pageCount?: number;
  status: BatchItemStatus;
  progress: number;
  stageText?: string;
  decodedLinesCount?: number;
  totalLinesCount?: number;
  result?: DocumentOCRResult;
  error?: string;
}
