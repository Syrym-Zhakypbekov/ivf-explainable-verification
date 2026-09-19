from __future__ import annotations

import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACT_MANIFEST.json"
OUT = ROOT / "dist" / "ivf-explainable-verification-v2.0.0.zip"
PREFIX = "ivf-explainable-verification-v2.0.0"
FIXED_TIME = (2026, 9, 19, 0, 0, 0)


def add_file(archive, path, arcname):
    info = zipfile.ZipInfo(arcname, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, path.read_bytes())


def main():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        add_file(archive, MANIFEST, f"{PREFIX}/{MANIFEST.name}")
        for item in payload["files"]:
            relative = Path(item["path"])
            add_file(archive, ROOT / relative, f"{PREFIX}/{relative.as_posix()}")
    print(f"OK ✓ {OUT.name} · {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
