"""Package report evidence and source with a SHA-256 file manifest."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = [
        "COURSE_PROJECT_REPORT.tex", "COURSE_PROJECT_REPORT.pdf",
        "COURSE_PROJECT_SLIDES.tex", "COURSE_PROJECT_SLIDES.pdf",
        "COURSE_PROJECT_REFERENCES.bib", "COURSE_PROJECT_SUMMARY.md",
        "COURSE_PROJECT_PRESENTATION_OUTLINE.md", "COURSE_PROJECT_VERIFICATION.md",
        "REPRODUCIBILITY.md", "REPRODUCIBILITY_ENVIRONMENT.json", "README.md", "requirements.txt",
        "icml2026.sty", "icml2026.bst", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty",
    ]
    files = [ROOT / name for name in names]
    for folder in ("src", "tests", "configs", "scripts", "results", "docs"):
        files.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and "archive" not in path.parts)
    files = sorted(set(files))
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = "".join(f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n"
                       for path in files)
    manifest_path = ROOT / "REPRODUCIBILITY_MANIFEST.sha256"
    manifest_path.write_text(manifest, encoding="utf-8", newline="\n")
    target = ROOT / "CS886_REPRODUCIBILITY.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(manifest_path, manifest_path.name)
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for path in files:
            packaged = archive.read(path.relative_to(ROOT).as_posix())
            if sha256(packaged).digest() != sha256(path.read_bytes()).digest():
                raise RuntimeError(f"Archive hash mismatch: {path}")
    print(f"Verified {len(files)} files in {target.name}, {target.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
