import pytest

from pipeline.training.ship_gate import (
    PROVEN_SHIP_MODEL,
    assert_shippable_checkpoint,
    decide_ship,
)


def test_assert_shippable_allows_handwritten_base() -> None:
    assert assert_shippable_checkpoint("microsoft/trocr-base-handwritten") == "microsoft/trocr-base-handwritten"
    assert assert_shippable_checkpoint(PROVEN_SHIP_MODEL) == PROVEN_SHIP_MODEL


def test_assert_shippable_rejects_stage1_scientific_run() -> None:
    with pytest.raises(ValueError, match="stage1"):
        assert_shippable_checkpoint("runs/base_iam_v1/best")
    with pytest.raises(ValueError, match="stage1"):
        assert_shippable_checkpoint("microsoft/trocr-base-stage1")


def evidence(report, cer=.048):
    return {**report, "measured": True, "sample_count": 100, "manifest_hash": "same", "checkpoint_hash": "weights", "baseline": {"measured": True, "sample_count": 100, "manifest_hash": "same", "cer": cer}}


def test_decide_ship_requires_strictly_lower_cer() -> None:
    report = {"checkpoint": PROVEN_SHIP_MODEL, "cer": 0.040, "num_beams": 1}
    assert decide_ship(evidence(report), baseline_cer=0.048)["promote"] is True
    worse = {"checkpoint": PROVEN_SHIP_MODEL, "cer": 0.050, "num_beams": 1}
    assert decide_ship(evidence(worse), baseline_cer=0.048)["promote"] is False


def test_decide_ship_uses_beams4_report_floor(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"cer": 0.04519364725550293}\n', encoding="utf-8")
    worse = {"checkpoint": PROVEN_SHIP_MODEL, "cer": 0.046, "num_beams": 4}
    assert decide_ship(evidence(worse,.04519364725550293), baseline_report_path=baseline)["promote"] is False
    better = {"checkpoint": PROVEN_SHIP_MODEL, "cer": 0.040, "num_beams": 4}
    assert decide_ship(evidence(better,.04519364725550293), baseline_report_path=baseline)["promote"] is True


def test_decide_ship_holds_beams4_without_baseline(tmp_path) -> None:
    report = {
        "checkpoint": "runs/base_handwritten_v1/best",
        "cer": 0.03,
        "num_beams": 4,
    }
    decision = decide_ship(
        report,
        output_dir=tmp_path,
        baseline_report_path=tmp_path / "missing_beams4.json",
    )
    assert decision["promote"] is False
    assert "missing" in decision["reason"]
    assert (tmp_path / "ship_decision.json").is_file()
