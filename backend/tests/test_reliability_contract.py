import asyncio
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from pydantic import ValidationError
from backend.app.schemas import RecognitionOptions, WordBox, FeedbackCorrectionPayload
from backend.app.config import Settings
from backend.app.routes.feedback import _persist_correction
from backend.app.routes.jobs import InMemoryJobStore
from backend.app.engine import InferenceEngine
from pipeline.training.ship_gate import decide_ship
from pipeline.training.release_registry import activate, rollback, checkpoint_digest


def test_mode_defaults_and_conflict():
    assert RecognitionOptions().processing_mode == 'local'
    with pytest.raises(ValidationError): RecognitionOptions(turbo=True)
    assert RecognitionOptions(processing_mode='cloud', turbo=True).processing_mode == 'cloud'


def test_unknown_confidence():
    assert WordBox(word_id='w', text='a', confidence=None, bbox=[0,0,.1,.1]).confidence is None


def test_missing_model_never_returns_text():
    engine=object.__new__(InferenceEngine)
    engine.model=None;engine.processor=None
    with pytest.raises(RuntimeError): engine.recognize_single_crop(None)


def test_local_never_invokes_cloud(monkeypatch):
    engine=InferenceEngine(execution_mode='mock')
    engine.settings=Settings(AZURE_OPENAI_ENDPOINT='https://example.test', AZURE_OPENAI_API_KEY='key', ENABLE_TURBO_MODE=True)
    cloud=Mock(side_effect=AssertionError('cloud called'))
    monkeypatch.setattr(engine,'_recognize_turbo_gen',cloud)
    with pytest.raises(Exception): list(engine._recognize_gen(b'invalid', 'x.png', RecognitionOptions()))
    cloud.assert_not_called()


def test_feedback_deduplicated_local_collect_only(tmp_path,monkeypatch):
    settings=Settings(FEEDBACK_MANIFEST_PATH=str(tmp_path/'manifest.jsonl'),FEEDBACK_CROPS_DIR=str(tmp_path/'crops'), FEEDBACK_STORAGE_BACKEND='azure_blob',AZURE_STORAGE_ACCOUNT_NAME='configured')
    monkeypatch.setattr('backend.app.routes.feedback._get_azure_sink', Mock(side_effect=AssertionError('cloud called')))
    monkeypatch.setattr('backend.app.routes.feedback.get_engine', Mock(side_effect=AssertionError('adaptation called')))
    payload=FeedbackCorrectionPayload(document_id='d',line_id='l',original_prediction='a',operator_correction='b',submission_id='same')
    a=_persist_correction(payload,settings);b=_persist_correction(payload,settings)
    assert a.feedback_id==b.feedback_id
    assert len((tmp_path/'manifest.jsonl').read_text().splitlines())==1
    assert a.confusion_pairs_updated==[]


@pytest.mark.asyncio
async def test_polling_never_executes_and_worker_executes_once(monkeypatch):
    recognize=Mock(return_value={'result':'ok'})
    monkeypatch.setattr('backend.app.routes.jobs.get_engine',lambda:SimpleNamespace(recognize=recognize))
    store=InMemoryJobStore(1,0)
    job=store.create_job('x',b'bytes')
    for _ in range(10): assert store.get_job(job)['status']=='QUEUED'
    assert recognize.call_count==0
    await asyncio.gather(store.execute_background_job(job),store.execute_background_job(job))
    assert recognize.call_count==1
    assert store._jobs[job]['file_bytes']==b''
    store.expire_jobs();assert job not in store._jobs


@pytest.mark.parametrize('report',[{'cer':.01}, {'cer':float('nan'),'measured':True,'sample_count':1,'manifest_hash':'a'}, {'cer':.01,'measured':True,'sample_count':0,'manifest_hash':'a'}])
def test_gate_missing_evidence_blocks(report):
    assert decide_ship(report)['promote'] is False


def test_activation_failure_preserves_state_and_rollback(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();(a/'weights.bin').write_bytes(b'a');(b/'weights.bin').write_bytes(b'b')
    registry=tmp_path/'active.json'
    activate(a,registry,{'promote':True,'checkpoint_hash':checkpoint_digest(a)},loader=lambda _:None)
    previous=registry.read_bytes()
    def fail(_): raise RuntimeError('cannot load')
    with pytest.raises(RuntimeError): activate(b,registry,{'promote':True,'checkpoint_hash':checkpoint_digest(b)},loader=fail)
    assert registry.read_bytes()==previous
    activate(b,registry,{'promote':True,'checkpoint_hash':checkpoint_digest(b)},loader=lambda _:None)
    assert rollback(registry,loader=lambda _:None)['active']==str(a)

@pytest.mark.asyncio
async def test_two_stream_subscribers_do_not_duplicate_inference(monkeypatch):
    import backend.app.routes.jobs as jobs
    recognize=Mock(return_value={'text':'done'})
    monkeypatch.setattr(jobs,'get_engine',lambda:SimpleNamespace(recognize=recognize))
    store=InMemoryJobStore(1,3600);monkeypatch.setattr(jobs,'job_store',store)
    job=store.create_job('x',b'bytes')
    async def consume(): return [event async for event in jobs._generate_sse_stream(job)]
    _,a,b=await asyncio.gather(store.execute_background_job(job),consume(),consume())
    assert recognize.call_count==1
    assert any('event: complete' in e for e in a)
    assert any('event: complete' in e for e in b)


def test_disk_failure_cannot_acknowledge_or_adapt(tmp_path,monkeypatch):
    settings=Settings(FEEDBACK_MANIFEST_PATH=str(tmp_path/'manifest.jsonl'))
    def fail(*args,**kwargs): raise OSError('disk full')
    monkeypatch.setattr('backend.app.routes.feedback._append_to_manifest',fail)
    payload=FeedbackCorrectionPayload(document_id='d',line_id='l',original_prediction='a',operator_correction='b')
    with pytest.raises(OSError): _persist_correction(payload,settings)


def test_release_rejects_mismatched_split_nan_and_clinical_gaps():
    good={'checkpoint':'microsoft/trocr-base-handwritten','cer':.03,'measured':True,'sample_count':10,'manifest_hash':'same','checkpoint_hash':'hash','baseline':{'measured':True,'sample_count':10,'manifest_hash':'same','cer':.04}}
    assert decide_ship(good)['promote']
    assert not decide_ship({**good,'manifest_hash':'different'})['promote']
    assert not decide_ship({**good,'cer':float('nan')})['promote']
    assert not decide_ship({**good,'domain':'clinical'})['promote']
    assert not decide_ship({**good,'failures':1})['promote']


def test_missing_scores_are_unknown():
    from backend.app.engine import _sequence_scores
    assert _sequence_scores(object(),SimpleNamespace(),2)==[None,None]
