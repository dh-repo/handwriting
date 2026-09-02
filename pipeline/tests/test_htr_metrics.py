from pipeline.training.metrics import cased_cer, cased_wer, exact_match, htr_metric_bundle, uncased_cer


def test_cased_cer_is_case_sensitive() -> None:
    assert cased_cer(["Hello"], ["hello"]) > 0.0
    assert uncased_cer(["Hello"], ["hello"]) == 0.0
    assert cased_cer(["abc"], ["abc"]) == 0.0


def test_wer_and_exact_match() -> None:
    assert cased_wer(["the cat"], ["the dog"]) > 0.0
    assert exact_match(["the cat"], ["the cat"]) == 1.0
    assert exact_match(["the cat"], ["The cat"]) == 0.0


def test_headline_bundle_does_not_lowercase() -> None:
    bundle = htr_metric_bundle(["IAM"], ["iam"])
    assert bundle["cer"] == cased_cer(["IAM"], ["iam"])
    assert bundle["cer"] > 0.0
    assert "uncased_cer" in bundle
    assert "wer" in bundle
    assert "exact_match" in bundle
