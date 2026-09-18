"""Create a shareable ZIP using an explicit allowlist; never include the manuscript."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    files = [root / name for name in ["README.md", "requirements.txt", "requirements-lock.txt",
             "config.json", ".gitignore", "run_pipeline.py", "predict_new_logs.py", "package_project.py",
             "spyder_permeability.py", "build_spyder_script.py", "SPYDER_README.md"]]
    for folder in ["permeability", "tests", ".github", "outputs"]:
        files.extend(p for p in (root / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    archive = root / "synthetic_permeability_github.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, Path("synthetic-permeability") / path.relative_to(root))
    print(f"Created {archive.name}: {len(files)} files, {archive.stat().st_size / 1e6:.1f} MB")
