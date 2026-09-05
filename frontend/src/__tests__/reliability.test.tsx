import { describe, it, expect, vi } from 'vitest';
import 'fake-indexeddb/auto';
import { ApiClient } from '../lib/apiClient';
import { saveDocument, listDocuments, deleteDocument, restoreDocument, pendingFeedback } from '../lib/documentStore';
import { SAMPLE_CLEAN_CURSIVE } from '../lib/sampleDocuments';
import { getConfidenceTier } from '../lib/colorUtils';

describe('Reliability contract', () => {
  it('never invents recognition when backend is unavailable even if fallback requested', async () => {
    vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));
    const client=new ApiClient({ baseUrl:'http://backend',enableFallback:true });
    await expect(client.recognizeBase64('abc')).rejects.toThrow();
  });
  it('persists failed feedback in the outbox without reporting success', async () => {
    vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));
    const client=new ApiClient({ baseUrl:'http://backend' });
    await expect(client.submitFeedback({ document_id:'test',line_id:'line',original_text:'old',corrected_text:'new',submission_id:'retry-me' })).rejects.toThrow();
    expect((await pendingFeedback()).find(p=>p.submission_id==='retry-me')).toBeTruthy();
  });
  it('stores edits and review state across reload and deletes saved documents', async () => {
    const doc=structuredClone(SAMPLE_CLEAN_CURSIVE);
    doc.document_id='saved-test';doc.pages.forEach(p=>{delete p.image_url;});
    doc.pages[0].lines[0].text='Corrected text';doc.pages[0].lines[0].reviewed=true;
    await saveDocument(doc);
    const record=(await listDocuments()).find(r=>r.id===doc.document_id)!;
    expect(restoreDocument(record).pages[0].lines[0].text).toBe('Corrected text');
    expect(record.document.pages[0].lines[0].reviewed).toBe(true);
    await deleteDocument(doc.document_id);
    expect((await listDocuments()).find(r=>r.id===doc.document_id)).toBeUndefined();
  });
  it('unknown confidence is reviewable',()=>expect(getConfidenceTier(null)).toBe('low'));
});

import React from 'react';
import { render, fireEvent, screen, waitFor, cleanup } from '@testing-library/react';
import { DocumentProvider, useDocumentContext } from '../context/DocumentContext';
import { LocalDocuments } from '../components/LocalDocuments';
import { InlineEditor } from '../components/InlineEditor';
import { exportDocumentAsTxt } from '../lib/exportUtils';
function ReviewHarness() {
  const {document,updateLineText}=useDocumentContext();
  return <><LocalDocuments />{document && <><input aria-label="Edit transcript" value={document.pages[0].lines[0].text} onChange={e=>updateLineText(document.pages[0].lines[0].line_id,e.target.value)} /><output>{exportDocumentAsTxt(document)}</output></>}</>;
}
it('edits, saves, resumes in a new provider, exports the edit, and deletes', async()=>{
  const doc=structuredClone(SAMPLE_CLEAN_CURSIVE);doc.document_id='resume-workflow';doc.is_mock=false;doc.is_demo=false;doc.pages.forEach(p=>delete p.image_url);
  const first=render(<DocumentProvider initialDocument={doc}><ReviewHarness /></DocumentProvider>);
  fireEvent.change(screen.getByLabelText('Edit transcript'),{target:{value:'Verified transcription'}});
  await waitFor(()=>expect(screen.getByText('Saved on this device')).toBeTruthy());
  first.unmount();
  render(<DocumentProvider><ReviewHarness /></DocumentProvider>);
  await waitFor(()=>expect(screen.getByRole('option',{name:doc.filename})).toBeTruthy());
  fireEvent.change(screen.getByLabelText('Resume saved document'),{target:{value:doc.document_id}});
  expect(screen.getByLabelText('Edit transcript')).toHaveValue('Verified transcription');
  expect(screen.getByText(/Verified transcription/, {selector:'output'})).toBeTruthy();
  fireEvent.click(screen.getByText('Delete local document'));
  await waitFor(()=>expect(screen.queryByLabelText('Edit transcript')).toBeNull());
  cleanup();
});
it('general review includes unknown scores without medical suggestions',()=>{
  const page=structuredClone(SAMPLE_CLEAN_CURSIVE.pages[0]);page.mean_confidence=null;page.lines.forEach(l=>{l.confidence=null;l.words.forEach(w=>w.confidence=null);});
  render(<InlineEditor page={page} documentId="demo" />);
  expect(screen.getAllByText(/Unknown/).length).toBeGreaterThan(0);
  expect(screen.queryByTestId('medical-autocomplete-popover')).toBeNull();
  cleanup();
});
