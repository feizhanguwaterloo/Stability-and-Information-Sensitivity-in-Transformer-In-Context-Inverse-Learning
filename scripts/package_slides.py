"""Package the standalone Beamer source and saved inputs for its figures."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = [
        "COURSE_PROJECT_SLIDES_V3.tex", "COURSE_PROJECT_SLIDES_V3.pdf",
        "COURSE_PROJECT_V3_REHEARSAL.md", "COURSE_PROJECT_V3_REVIEW.md",
        "COURSE_PROJECT_PRESENTATION_OUTLINE.md", "requirements.txt",
        "scripts/build_slide_assets.py", "scripts/build_report_assets.py",
        "scripts/package_slides.py",
        "results/transformer_stability/recovery_summary.csv",
        "results/transformer_stability/stability_summary.csv",
        "results/leverage_stability/identifying_pair_summary.csv",
        "results/leverage_stability/identifying_pair_raw.csv",
        "results/leverage_stability/matched_summary.csv",
        "results/task_scaling/risks_summary.csv",
        "results/leverage_stability/norm_bounds.csv",
        "results/presentation/pair_response_check.tex",
    ]
    names += [f"results/presentation/{name}.pdf" for name in
              ("architecture", "identifying_response", "matched_response", "transfer")]
    target = ROOT / "CS886_BEAMER_SOURCE.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Beamer archive CRC verification failed")
        for name in names:
            if archive.read(name) != (ROOT / name).read_bytes():
                raise RuntimeError(f"Beamer archive content mismatch: {name}")
    print(f"Verified {len(names)} files in {target.name}")


if __name__ == "__main__":
    main()
