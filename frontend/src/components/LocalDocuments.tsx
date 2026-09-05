'use client';
import React, { useEffect, useRef, useState } from 'react';
import { useDocumentContext } from '../context/DocumentContext';
import { saveDocument, listDocuments, deleteDocument, restoreDocument, pendingFeedback, removeFeedback, SavedDocument } from '../lib/documentStore';
import { apiClient } from '../lib/apiClient';
export function LocalDocuments() {
  const { document, setDocument, isProcessing } = useDocumentContext();
  const [status, setStatus] = useState('');
  const [records, setRecords] = useState<SavedDocument[]>([]);
  const [feedback, setFeedback] = useState('');
  const chain = useRef(Promise.resolve());
  const revision = useRef(0);
  useEffect(() => { listDocuments().then(setRecords).catch(() => setStatus('Local storage unavailable — export to save')); }, []);
  useEffect(() => {
    const current = ++revision.current;
    if (!document || isProcessing || document.is_demo || document.is_mock) return;
    setStatus('Saving on this device…');
    {
      chain.current = chain.current.catch(() => {}).then(async () => {
        await saveDocument(document);
        if (revision.current === current) setStatus('Saved on this device');
        setRecords(await listDocuments());
      }).catch(() => { if (revision.current === current) setStatus('Local save failed — export your edits'); });
    }
  }, [document, isProcessing]);
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => { if (status.startsWith('Saving') || status.startsWith('Local save failed')) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [status]);
  async function retry() {
    setFeedback('Submitting corrections…');
    try { for (const p of await pendingFeedback()) { await apiClient.submitFeedback(p); await removeFeedback(p.submission_id!); } setFeedback('Corrections submitted; automatic learning is disabled'); }
    catch { setFeedback('Corrections remain pending on this device'); }
  }
  return <section className="px-4 py-2 border-b border-zinc-800 text-xs text-zinc-300 flex flex-wrap gap-3 items-center" aria-label="Local documents">
    <span role="status">{status}</span><span>Active review: {Math.round((document?.active_review_ms || 0) / 1000)}s</span>
    <span>{document?.is_demo ? 'Demo — not a recognition result' : document ? `${document.processing_location || 'local'} · ${document.model_id || document.engine_used || 'engine unknown'}${document.incomplete ? ' · Incomplete — review before exporting' : ''}` : 'Local processing by default; select Cloud explicitly to send a document to Azure.'}</span>
    <select aria-label="Resume saved document" value="" onChange={e => { const r = records.find(r => r.id === e.target.value); if (r) setDocument(restoreDocument(r)); }} className="bg-zinc-900 p-1">
      <option value="">Resume a saved document</option>{records.map(r => <option key={r.id} value={r.id}>{r.document.filename}</option>)}
    </select>
    {document && <button onClick={async () => { revision.current++; await chain.current; await deleteDocument(document.document_id); setDocument(null); setRecords(await listDocuments()); setStatus('Document deleted from this device'); }}>Delete local document</button>}
    <button onClick={retry}>Retry pending feedback</button><span>{feedback}</span>
    <span className="text-zinc-400">Browser storage can be cleared. Export a portable backup.</span>
  </section>;
}
