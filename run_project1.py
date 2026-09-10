"""Reproduce the three experiments and build the webpage from their records."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import PIL
from proj1 import process

ROOT = Path(__file__).resolve().parent


def main():
    import os
    os.chdir(ROOT)
    provided = sorted(p for p in Path("proj1_data").iterdir() if p.suffix.lower() in (".jpg", ".tif", ".tiff"))
    sources = json.loads(Path("additional_sources.json").read_text())
    additional = [Path("additional_data") / source["filename"] for source in sources]
    if len(provided) != 14 or not all(p.exists() for p in additional):
        raise SystemExit("Expected 14 supplied images in proj1_data/ and 3 extras; run download_examples.py first.")
    options = dict(metric="ncc", feature="raw", method="pyramid", radius=15,
                   refine_radius=3, crop=0.12, coarsest_size=350, max_samples=0)
    experiments = (("single", [p for p in provided if p.suffix.lower() == ".jpg"], dict(options, method="single")),
                   ("pyramid", provided + additional, options),
                   ("gradient", provided + additional, dict(options, feature="gradient")))
    for name, paths, settings in experiments:
        output = Path("outputs") / name
        records = [process(path, output, **settings) for path in paths]
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    metadata = {"run_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
                "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform(),
                "machine": platform.machine(), "timing": "perf_counter; sequential execution; total includes reading, alignment, composition and JPEG writing",
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provided + additional}}
    Path("outputs/run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    from build_project1_page import main as build_page
    build_page()


if __name__ == "__main__":
    main()
