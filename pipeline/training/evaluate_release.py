"""Manual release evaluation: baseline and candidate run against identical frozen images."""
import argparse
import json
from pathlib import Path
from pipeline.evaluation.reliable_benchmark import run
from pipeline.training.release_registry import checkpoint_digest
from pipeline.training.ship_gate import decide_ship, load_lasa_catalog, audit_lasa_safety


def evaluate_release(candidate, baseline, holdout, output, clinical_holdout=None):
    Path(output).mkdir(parents=True, exist_ok=False)
    baseline_report = run(Path(holdout), baseline)
    candidate_report = run(Path(holdout), candidate)
    candidate_report.update(baseline=baseline_report, checkpoint_hash=checkpoint_digest(candidate), domain='general')
    audit = None
    if clinical_holdout:
        import re
        clinical = run(Path(clinical_holdout), candidate)
        pairs = load_lasa_catalog()
        refs = [row['text'] for row in clinical['predictions']]
        hyps = [row['prediction'] for row in clinical['predictions']]
        covered = sum(any(re.search(r'\b'+re.escape(term)+r'\b', ref, re.I) and not re.search(r'\b'+re.escape(other)+r'\b', ref, re.I) for ref in refs) for a,b in pairs for term,other in [(a,b),(b,a)])
        audit = audit_lasa_safety(refs, hyps, lasa_pairs=pairs)
        candidate_report.update(domain='clinical', clinical_evaluation={
            'measured': True, 'sample_count': len(refs), 'failures': clinical['failures'],
            'manifest_hash': clinical['manifest_hash'], 'checkpoint_hash': candidate_report['checkpoint_hash'],
            'required_directions': len(pairs)*2, 'covered_directions': covered})
    result = decide_ship(candidate_report, output_dir=output, max_cer_regression=0.05, lasa_audit=audit)
    Path(output, 'evaluation.json').write_text(json.dumps(candidate_report, indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',required=True);p.add_argument('--holdout',required=True);p.add_argument('--output',required=True);p.add_argument('--clinical-holdout');a=p.parse_args()
    print(evaluate_release(a.candidate,a.baseline,a.holdout,a.output,a.clinical_holdout))
