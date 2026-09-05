import type { DocumentOCRResult, FeedbackSubmissionRequest } from '../types/ocr';
export type SavedDocument = { id: string; document: DocumentOCRResult; images: (Blob | null)[]; original?: Blob; updated: number; revisions?: DocumentOCRResult[] };
async function db(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('handwriting-workspace', 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('documents', { keyPath: 'id' });
      request.result.createObjectStore('feedback', { keyPath: 'submission_id' });
      request.result.createObjectStore('originals');
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function transaction<T>(store: string, mode: IDBTransactionMode, action: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const database = await db();
  return new Promise((resolve, reject) => {
    const tx = database.transaction(store, mode);
    const request = action(tx.objectStore(store));
    tx.oncomplete = () => { database.close(); resolve(request.result); };
    tx.onerror = tx.onabort = () => { database.close(); reject(tx.error || request.error || new Error('Local storage failed')); };
  });
}
export async function rememberOriginal(file: File): Promise<string> { const key = crypto.randomUUID(); await transaction('originals', 'readwrite', s => s.put(file, key)); return key; }
export async function saveDocument(document: DocumentOCRResult): Promise<void> {
  const old = await transaction<SavedDocument | undefined>('documents', 'readonly', s => s.get(document.document_id));
  const images = await Promise.all(document.pages.map(async (p, i) => {
    if (old?.images[i]) return old.images[i];
    if (!p.image_url) return null;
    const response = await fetch(p.image_url);
    if (!response.ok) throw new Error('Could not save document image');
    return response.blob();
  }));
  const copy = structuredClone(document);
  copy.pages.forEach(p => { delete p.image_url; delete p.image_data_url; });
  const original = await transaction<Blob | undefined>('originals', 'readonly', s => s.get(document.original_key || document.filename));
  await transaction('documents', 'readwrite', s => s.put({ id: document.document_id, document: copy, images, original: original || old?.original, updated: Date.now(), revisions: [...(old?.revisions || []), ...(old ? [old.document] : [])].slice(-20) }));
}
export const listDocuments = () => transaction<SavedDocument[]>('documents', 'readonly', s => s.getAll());
export async function deleteDocument(id: string) {
  const record = await transaction<SavedDocument | undefined>('documents', 'readonly', s => s.get(id));
  await transaction('documents', 'readwrite', s => s.delete(id));
  if (record) await transaction('originals', 'readwrite', s => s.delete(record.document.original_key || record.document.filename));
  for (const feedback of await pendingFeedback()) if (feedback.document_id === id) await removeFeedback(feedback.submission_id!);
}
export function restoreDocument(record: SavedDocument): DocumentOCRResult {
  const doc = structuredClone(record.document);
  doc.pages.forEach((p, i) => { if (record.images[i]) p.image_url = URL.createObjectURL(record.images[i]!); });
  return doc;
}
export const queueFeedback = (payload: FeedbackSubmissionRequest) => transaction('feedback', 'readwrite', s => s.put(payload));
export const pendingFeedback = () => transaction<FeedbackSubmissionRequest[]>('feedback', 'readonly', s => s.getAll());
export const removeFeedback = (id: string) => transaction('feedback', 'readwrite', s => s.delete(id));
