"""Download the three original LoC TIFF scans recorded in the source manifest."""
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    folder = root / "additional_data"
    folder.mkdir(exist_ok=True)
    for source in json.loads((root / "additional_sources.json").read_text()):
        target = folder / source["filename"]
        expected = source.get("sha256")
        if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"Verified {target.name}")
            continue
        temporary = target.with_suffix(".tif.part")
        print(f"Downloading {source['title']}", flush=True)
        request = urllib.request.Request(source["download_url"], headers={"User-Agent": "CS180-course-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if expected and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {target.name}")
        temporary.replace(target)


if __name__ == "__main__":
    main()
