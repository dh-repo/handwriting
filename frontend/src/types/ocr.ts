/**
 * Unified OCR and Document Data Models for Handwriting Recognition
 * Aligned with backend FastAPI Pydantic schemas (PROJECT.md § Interface Contracts).
 */

export type BoundingBoxTuple = [number, number, number, number]; // [ymin, xmin, ymax, xmax] in [0, 1]

export interface NormalizedBoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

export type PolygonPoint = [number, number]; // [x, y] in normalized [0, 1] coordinates

export interface WordCandidate {
  text: string;
  confidence: number;
}

export interface WordToken {
  word_id: string;
  text: string;
  original_text?: string;
  confidence: number; // 0.0 to 1.0
  bbox: BoundingBoxTuple; // [ymin, xmin, ymax, xmax]
  polygon?: PolygonPoint[];
  alternatives?: (WordCandidate | string)[];
  candidate_tokens?: WordCandidate[];
  is_low_confidence?: boolean;
  is_edited?: boolean;
  is_proper_noun?: boolean;
}

export interface LineItem {
  line_id: string;
  line_index?: number;
  line_number?: number;
  text: string;
  original_text?: string;
  confidence: number; // 0.0 to 1.0
  bbox: BoundingBoxTuple; // [ymin, xmin, ymax, xmax]
  polygon?: PolygonPoint[];
  words: WordToken[];
  is_edited?: boolean;
}

export interface PageResult {
  page_number: number;
  width: number;
  height: number;
  full_text: string;
  mean_confidence: number;
  confidence?: number;
  lines: LineItem[];
  image_url?: string;
  image_data_url?: string;
  title?: string;
}

export interface DocumentOCRResult {
  document_id: string;
  filename: string;
  mime_type?: string;
  total_pages: number;
  pages: PageResult[];
  full_text?: string;
  mean_confidence?: number;
  overall_confidence?: number;
  processing_time_ms: number;
  model_version?: string;
  preprocessing_flags?: Record<string, unknown>;
  is_mock?: boolean;
  signature_reviews?: SignatureReviewRecord[];
  engine_used?: string;
}

export type SignatureDecision = 'pending' | 'accepted' | 'rejected';

export interface SignatureReviewRecord {
  page_number: number;
  line_id: string;
  kind: 'sign_off_line' | 'witness_mark' | 'handwritten_mark';
  decision: SignatureDecision;
  decided_at?: string;
}

// Alias for RecognitionResponse matching backend schema
export type RecognitionResponse = DocumentOCRResult;
export type PageOCRResult = PageResult;
export type LineOCRResult = LineItem;
export type WordOCRResult = WordToken;

export interface LowConfidenceWordItem {
  page_index: number;
  line_id: string;
  word_id: string;
  text: string;
  original_text: string;
  confidence: number;
  bbox: BoundingBoxTuple;
  page_image_url?: string;
  alternatives?: (WordCandidate | string)[];
  is_proper_noun?: boolean;
}

export type ExportFormat = 'json' | 'txt' | 'csv';

export type ConfidenceTier = 'high' | 'medium' | 'low';

export interface RecognitionOptions {
  deskew?: boolean;
  enhance_contrast?: boolean;
  binarization_method?: 'sauvola' | 'otsu' | 'none' | string;
  extract_words?: boolean;
  dpi?: number;
  beam_width?: number;
  rescore?: boolean;
  adaptive?: boolean;
  model_type?: string;
  turbo?: boolean;
}

export interface RecognizeJsonRequest {
  file_base64: string;
  filename?: string;
  options?: RecognitionOptions;
  sample_id?: string;
  model_type?: string;
}

export type JobStatusType = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export enum JobStatusEnum {
  QUEUED = 'QUEUED',
  PROCESSING = 'PROCESSING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
}

export interface JobSubmissionResponse {
  job_id: string;
  status: JobStatusEnum | JobStatusType;
  filename: string;
  created_at: string;
}

export interface JobStatusResponse {
  job_id: string;
  filename: string;
  status: JobStatusEnum | JobStatusType;
  progress: number;
  result?: DocumentOCRResult | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface HealthResponse {
  status: string;
  device: string;
  version: string;
  memory_usage_mb?: number | null;
  loaded_models?: string[] | null;
  mps_available?: boolean | null;
  execution_mode?: string | null;
  rescorer_active: boolean;
  timestamp: string;
}

export interface ApiErrorResponse {
  error: string;
  detail?: string;
  details?: string;
  code?: string;
  timestamp?: string;
}

export type ErrorResponse = ApiErrorResponse;

export type SSEEventType = 'progress' | 'complete' | 'error' | 'stage' | 'heartbeat' | 'message';

export interface SSEEvent<T = unknown> {
  event: SSEEventType | string;
  data: T;
  id?: string;
  retry?: number;
}

export type StreamingEvent = SSEEvent;

export interface FeedbackSubmissionRequest {
  document_id: string;
  page_number?: number;
  line_id: string;
  word_id?: string;
  original_prediction?: string;
  operator_correction?: string;
  original_text?: string;
  corrected_text?: string;
  confidence?: number;
  bbox?: BoundingBoxTuple; // [ymin, xmin, ymax, xmax]
  line_crop_base64?: string; // data:image/png;base64,...
  timestamp?: string; // ISO-8601 UTC
}

export interface FeedbackSubmissionResponse {
  status: 'persisted' | 'queued' | 'acknowledged';
  feedback_id: string;
  document_id?: string;
  line_id?: string;
  manifest_path?: string;
  crop_path?: string;
  confusion_pairs_count?: number;
  confusion_pairs_updated?: Array<{
    source: string;
    target: string;
    old_cost?: number;
    new_cost?: number;
  }>;
  timestamp: string;
}

export type FeedbackSyncStatus = 'idle' | 'debouncing' | 'syncing' | 'synced' | 'error';

