"""Package FINAL without rewriting any V3 or experimental artifact."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = [
        "COURSE_PROJECT_SLIDES_FINAL.tex", "COURSE_PROJECT_SLIDES_FINAL.pdf",
        "COURSE_PROJECT_FINAL_NUMBERS.tex", "COURSE_PROJECT_FINAL_CHECKS.json",
        "COURSE_PROJECT_FINAL_REHEARSAL.md", "COURSE_PROJECT_FINAL_CHANGELOG.md",
        "scripts/verify_final_presentation.py", "scripts/package_final_slides.py",
        "results/leverage_stability/identifying_pair_raw.csv",
        "results/leverage_stability/identifying_pair_summary.csv",
        "results/leverage_stability/norm_bounds.csv",
    ]
    names += [f"results/presentation/{name}.pdf" for name in
              ("architecture", "identifying_response", "matched_response", "transfer")]
    target = ROOT / "CS886_BEAMER_FINAL_SOURCE.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for name in names:
            assert archive.read(name) == (ROOT / name).read_bytes(), name
    print(f"Verified {len(names)} files in {target.name}")


if __name__ == "__main__":
    main()
