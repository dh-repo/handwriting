import json
from pathlib import Path

from pipeline.training.eval import write_test_report, write_worst_tsv


def test_write_test_report_schema(tmp_path: Path) -> None:
    path = write_test_report(
        tmp_path / "test_report.json",
        split_name="Teklia/IAM-line test",
        checkpoint="runs/base_iam_v1/best",
        checkpoint_hash="abc123",
        cer=0.041,
        wer=0.12,
        uncased_cer=0.03,
        exact_match=0.55,
        beams=4,
        torch_version="2.13.0",
        transformers_version="5.0.0",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["split_name"] == "Teklia/IAM-line test"
    assert data["cer"] == 0.041
    assert data["wer"] == 0.12
    assert data["num_beams"] == 4
    assert data["checkpoint_hash"] == "abc123"
    assert "date" in data
    assert data["torch_version"] == "2.13.0"
    assert data["claim"] == "Teklia-pack CER, not paper IAM 3.42%"


def test_write_worst_tsv(tmp_path: Path) -> None:
    path = write_worst_tsv(
        tmp_path / "worst100.tsv",
        rows=[
            {"image_path": "a.png", "reference": "Hello", "hypothesis": "Hallo", "cer": 0.2},
            {"image_path": "b.png", "reference": "Cat", "hypothesis": "Cat", "cer": 0.0},
        ],
        n=1,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("image_path")
    assert "a.png" in lines[1]
    assert len(lines) == 2
