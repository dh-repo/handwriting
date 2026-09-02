from pathlib import Path

from pipeline.training.dataset import OCRDataset


def test_from_manifest_filters_to_requested_categories(tmp_path: Path) -> None:
    manifest = tmp_path / "mix.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"image_path":"a.png","text":"hello line","sample_id":"1","writer_id":"w1","category":"general_cursive_line"}',
                '{"image_path":"b.png","text":"word","sample_id":"2","writer_id":"w2","category":"in_the_wild_word"}',
                '{"image_path":"c.png","text":"old deed","sample_id":"3","writer_id":"w3","category":"historical_line"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ds = OCRDataset.from_manifest(
        manifest,
        categories=["general_cursive_line", "historical_line"],
        is_training=False,
    )
    texts = [ds.samples[i][1] for i in range(len(ds))]
    assert texts == ["hello line", "old deed"]
