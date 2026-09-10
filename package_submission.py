"""Build an explicit, image-free Gradescope archive."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = ["proj1.py", "test_proj1.py", "run_project1.py", "build_project1_page.py",
         "download_examples.py", "package_submission.py", "requirements.txt",
         "additional_sources.json", "README.md", "website_url.txt", "proj1.html", "proj1.css", "style.css"]


def main():
    folder = ROOT / "submission"
    folder.mkdir(exist_ok=True)
    target = folder / "project1-code.zip"
    records = [ROOT / f"assets/proj1/{kind}-results.json" for kind in ("single", "pyramid", "gradient")]
    records += [ROOT / "assets/proj1/run-metadata.json", ROOT / "assets/proj1/offsets.csv"]
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
        for path in records:
            archive.write(path, "results/" + path.name)
    with ZipFile(target) as archive:
        assert all(Path(name).suffix in (".py", ".md", ".txt", ".json", ".csv", ".html", ".css") for name in archive.namelist())
        assert archive.testzip() is None
    print(f"Created {target} ({target.stat().st_size:,} bytes), with no images.")


if __name__ == "__main__":
    main()
