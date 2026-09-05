"""Prevent exact frozen holdout images from entering correction/training collections."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from PIL import Image


def pixel_digest(image):
    rgb=image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


@lru_cache(maxsize=32)
def _manifest_pixels(path, mtime):
    manifest=Path(path)
    data=json.loads(manifest.read_text())
    if data.get('purpose')!='evaluation_only': return set()
    result=set()
    for row in data['rows']:
        image_path=manifest.parent/row['image']
        if image_path.is_file():
            with Image.open(image_path) as image: result.add(pixel_digest(image))
    return result


def is_holdout_image(image):
    digest=pixel_digest(image)
    for manifest in Path('benchmarks').glob('*/manifest.json'):
        if digest in _manifest_pixels(str(manifest),manifest.stat().st_mtime_ns): return True
    return False
