from pathlib import Path

import pytest
import yaml

from pipeline.training.hf_train import load_yaml_config, run_training


def test_yaml_configs_exist_and_disable_private_mix() -> None:
    root = Path("configs")
    for name in ("debug32.yaml", "base_iam.yaml", "large_iam.yaml", "base_handwritten.yaml"):
        cfg = load_yaml_config(root / name)
        assert cfg["mix_private"] is False
        assert int(cfg["generation_max_length"]) == 128
        assert int(cfg["generation_num_beams"]) == 1


def test_base_iam_starts_from_stage1() -> None:
    cfg = yaml.safe_load(Path("configs/base_iam.yaml").read_text(encoding="utf-8"))
    assert cfg["model_name_or_path"] == "microsoft/trocr-base-stage1"
    assert int(cfg["generation_num_beams"]) == 1
    assert int(cfg["test_num_beams"]) == 4
    assert cfg["test_dir"] == "data/iam_line/test"


def test_base_handwritten_is_loop_rehearsal_not_stage1() -> None:
    cfg = yaml.safe_load(Path("configs/base_handwritten.yaml").read_text(encoding="utf-8"))
    assert cfg["model_name_or_path"] == "microsoft/trocr-base-handwritten"
    assert float(cfg["learning_rate"]) <= 2.0e-5
    assert float(cfg["num_train_epochs"]) <= 2
    assert cfg["mix_private"] is False
    assert int(cfg["test_num_beams"]) == 4


def test_run_training_rejects_device_map_auto() -> None:
    with pytest.raises(ValueError, match="device_map"):
        run_training({"device_map": "auto", "mix_private": False})


def test_run_training_rejects_private_mix_until_iam_green() -> None:
    with pytest.raises(ValueError, match="private mix"):
        run_training({"mix_private": True})
