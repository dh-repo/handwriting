"""Freeze a public held-out line set and evaluate without synthetic substitutions.

python -m pipeline.evaluation.reliable_benchmark freeze --output benchmarks/iam
python -m pipeline.evaluation.reliable_benchmark run --output benchmarks/iam
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze(output: Path, limit=100):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    source = 'Teklia/IAM-line'
    revision = HfApi().dataset_info(source).sha
    if (output / 'manifest.json').exists():
        raise ValueError('Frozen manifest already exists; use a new directory')
    dataset = load_dataset(source, revision=revision, split='test')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(dataset):
        if i >= limit: break
        path = output / f'{i:05d}.png'
        row['image'].convert('RGB').save(path)
        label = row.get('text', row.get('transcription'))
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Missing real transcription')
        rows.append({'image': path.name, 'text': label, 'sha256': digest(path)})
    if not rows: raise ValueError('Empty holdout')
    manifest = {'purpose': 'evaluation_only', 'source': source, 'revision': revision, 'split': 'test', 'scope': 'line', 'rows': rows}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def run(output: Path, checkpoint='microsoft/trocr-base-handwritten', beams=4):
    import torch
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from pipeline.training.metrics import cased_cer, htr_metric_bundle
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('purpose') != 'evaluation_only' or not manifest['rows']:
        raise ValueError('A nonempty frozen evaluation manifest is required')
    revision = HfApi().model_info(checkpoint).sha if not Path(checkpoint).exists() else None
    processor = TrOCRProcessor.from_pretrained(checkpoint, revision=revision)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, revision=revision)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device).eval()
    refs, hyps, details = [], [], []
    start = time.perf_counter()
    for row in manifest['rows']:
        path = output / row['image']
        if digest(path) != row['sha256']: raise ValueError('Holdout content changed')
        t = time.perf_counter()
        error = None
        try:
            pixels = processor(images=Image.open(path).convert('RGB'), return_tensors='pt').pixel_values.to(device)
            with torch.inference_mode(): ids = model.generate(pixels, num_beams=beams, max_new_tokens=128)
            hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception as exc:
            hyp, error = '', str(exc)
        refs.append(row['text']); hyps.append(hyp)
        details.append({**row, 'prediction': hyp, 'error': error, 'latency_ms': (time.perf_counter()-t)*1000, 'cer': cased_cer([row['text']], [hyp])})
    report = {**htr_metric_bundle(refs, hyps), 'checkpoint': checkpoint, 'model_revision': revision,
              'manifest_hash': digest(manifest_path), 'scope': 'line', 'sample_count': len(refs), 'measured': True,
              'device': device, 'num_beams': beams, 'max_new_tokens': 128, 'failures': sum(x['error'] is not None for x in details),
              'elapsed_seconds': time.perf_counter()-start, 'predictions': details, 'worst_examples': sorted(details, key=lambda x:x['cer'], reverse=True)[:20]}
    path = output / f'report-{time.time_ns()}.json'
    path.write_text(json.dumps(report, indent=2)); print(path)
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action', choices=['freeze','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=100);p.add_argument('--checkpoint',default='microsoft/trocr-base-handwritten')
    a=p.parse_args()
    freeze(a.output,a.limit) if a.action=='freeze' else run(a.output,a.checkpoint)
