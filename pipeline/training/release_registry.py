"""Manual, load-validated checkpoint activation and rollback. Never called by training."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def checkpoint_digest(path):
    root=Path(path)
    if not root.is_dir(): raise ValueError('Activation requires an immutable local checkpoint directory')
    h=hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in {'ship_decision.json', 'evaluation.json'}:
            h.update(str(p.relative_to(root)).encode())
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def activate(checkpoint, registry, decision, loader=None):
    checkpoint=str(Path(checkpoint).resolve()); registry=Path(registry)
    if not decision.get('promote') or decision.get('checkpoint_hash') != checkpoint_digest(checkpoint):
        raise ValueError('Passing evaluation for these exact checkpoint bytes is required')
    if loader is None:
        from pipeline.training.model_contract import load_htr_model, load_htr_processor
        def loader(path):
            model=load_htr_model(path);load_htr_processor(path);model.eval()
    loader(checkpoint)  # No state change if validation/loading fails.
    registry.parent.mkdir(parents=True,exist_ok=True)
    previous=json.loads(registry.read_text()) if registry.exists() else None
    state={'active':checkpoint,'sha256':decision['checkpoint_hash'],'previous':previous}
    temp=registry.with_suffix('.tmp')
    with temp.open('w') as f: json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,registry)
    return state


def rollback(registry, loader=None):
    state=json.loads(Path(registry).read_text());previous=state.get('previous')
    if not previous: raise ValueError('No previous checkpoint')
    return activate(previous['active'],registry,{'promote':True,'checkpoint_hash':previous['sha256']},loader)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['activate','rollback']);p.add_argument('--registry',default='checkpoints/active.json');p.add_argument('--checkpoint');p.add_argument('--decision');a=p.parse_args()
    if a.action=='rollback': rollback(a.registry)
    else: activate(a.checkpoint,a.registry,json.loads(Path(a.decision).read_text()))
