"""
Deploy to the Hugging Face Space
================================
Syncs this working tree to https://huggingface.co/spaces/aynaval2003/namma-route

Why a script instead of `git push`: the Space was built by uploading files with
`huggingface_hub`, so its git history is unrelated to this repository's. A plain push
is rejected, and a force push would replace the Space's history — including the
`.gitattributes` that puts `yolov8n.pt` and the `.pkl` models in Git LFS. `upload_folder`
respects LFS, matches how the Space was created, and can delete stale files in the
same commit.

Deleting matters here. The Space currently carries a 9,405-row `learning_log.csv` in
which every "prediction" equals the ground-truth value. Uploading over the top would
leave it in place and the dashboard would go on reporting the ~95% accuracy that came
from it, so it is removed explicitly.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # or: export HF_TOKEN=hf_...
    python3 deploy_to_hf.py --dry-run
    python3 deploy_to_hf.py
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ID = "aynaval2003/namma-route"
REPO_TYPE = "space"
ROOT = Path(__file__).resolve().parent

# Never uploaded: local-only, secret, or runtime state the app regenerates.
IGNORE_PATTERNS = [
    ".git/*", ".git*/**",
    "**/__pycache__/**", "**/*.pyc",
    ".env", "*.log",
    "venv/**", ".venv/**", "env/**",
    "**/.DS_Store",
    "*.pptx",
    # Runtime state — the API recreates the log on boot and alerts are written by the
    # simulator. Shipping them would reinstate exactly the stale data being removed.
    "data/processed/learning_log.csv",
    "data/active_alerts.json",
    # Redundant copy of data/raw/astram_events.csv.
    "Astram event data_anonymized*",
    # Local scratch clone of the Space, if present.
    "hfspace/**",
]

# Removed from the Space in the same commit.
DELETE_PATTERNS = [
    "data/processed/learning_log.csv",       # 9,405 poisoned rows
    "data/active_alerts.json",               # 882 KB stale base64 frame
    "scratch_acc.py",                        # crashes on import; removed from the repo
    "data/processed/models/forecasting_engine.pkl",   # orphan, never loaded
    "Astram event data_anonymized*",         # duplicate of data/raw/astram_events.csv
]


def preflight_requirements() -> list:
    """Check requirements.txt actually installs on the Dockerfile's Python, for Linux.

    Exists because a deploy already failed this way: the pins were refreshed from a
    Python 3.14 development environment while the image was still built on
    python:3.10-slim, and pandas 3.x requires >=3.11. Nothing local catches that —
    the packages import fine on the developer's machine, and the container installs
    them before it runs a line of project code, so the first sign of trouble was a
    red build on a public demo. pip can resolve for another interpreter and platform
    without installing, which turns a ten-minute failed build into a two-second check.

    Returns a list of problems; empty means good.
    """
    dockerfile = ROOT / "Dockerfile"
    reqs = ROOT / "requirements.txt"
    if not dockerfile.exists() or not reqs.exists():
        return []

    m = re.search(r"^FROM\s+python:(\d+\.\d+)", dockerfile.read_text(), re.MULTILINE)
    if not m:
        return []
    pyver = m.group(1)

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps",
             "--only-binary=:all:", "--python-version", pyver,
             # opencv ships manylinux_2_17/2014 wheels, torch ships 2_28 — a single
             # platform tag rejects one or the other, so offer all the common ones.
             "--platform", "manylinux_2_17_x86_64",
             "--platform", "manylinux2014_x86_64",
             "--platform", "manylinux_2_28_x86_64",
             "--target", tmp, "-r", str(reqs)],
            capture_output=True, text=True, timeout=600,
        )
    if proc.returncode == 0:
        print(f"  pre-flight: requirements.txt resolves on Python {pyver} / linux x86_64")
        return []
    bad = [l.strip() for l in (proc.stdout + proc.stderr).splitlines()
           if "No matching distribution" in l or "Could not find a version" in l]
    return bad or [f"pip could not resolve requirements.txt for Python {pyver}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded and deleted, then exit.")
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    problems = preflight_requirements()
    if problems:
        print("\nPRE-FLIGHT FAILED — requirements.txt will not install in the container:")
        for b in problems:
            print(f"   {b}")
        print("\nFix the pins or the Dockerfile base image before deploying; this is")
        print("exactly the failure that breaks the build after the upload has landed.")
        sys.exit(1)

    if args.dry_run:
        import fnmatch
        print(f"Target Space : {args.repo_id}")
        print(f"Source       : {ROOT}\n")

        def ignored(rel: str) -> bool:
            for pat in IGNORE_PATTERNS:
                if fnmatch.fnmatch(rel, pat):
                    return True
                prefix = pat.split("*")[0].rstrip("/")
                if prefix and rel.startswith(prefix + "/"):
                    return True
                if "/" in rel and fnmatch.fnmatch(Path(rel).name, pat):
                    return True
            return False

        files = []
        for p in ROOT.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if ignored(rel):
                continue
            files.append((rel, p.stat().st_size))
        files.sort()
        total = sum(s for _, s in files)
        print(f"WOULD UPLOAD {len(files)} files ({total / 1e6:.1f} MB):")
        for rel, size in files:
            print(f"   {size:>10,}  {rel}")
        print("\nWOULD DELETE from the Space:")
        for pat in DELETE_PATTERNS:
            print(f"   {pat}")
        print("\nNothing was sent. Re-run without --dry-run to deploy.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed.  pip install huggingface_hub")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        api.repo_info(repo_id=args.repo_id, repo_type=REPO_TYPE)
    except Exception as exc:
        sys.exit(f"Cannot reach the Space (are you logged in?): {exc}")

    print(f"Uploading to {args.repo_id} ...")
    commit = api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        ignore_patterns=IGNORE_PATTERNS,
        delete_patterns=DELETE_PATTERNS,
        commit_message="Remove target leakage; rebuild forecasting on an honest estimator",
        commit_description=(
            "Severity was `priority`, an administrative corridor flag (99.84% "
            "determined), so the reported 91.4% measured nothing. It is now derived "
            "from the observed outcome and scores 51.5% against a 40.0% baseline. "
            "Duration and severity come from one empirical estimator that reports a "
            "calibrated interval and its sample size. Retraining from officer feedback "
            "is implemented. The poisoned learning log is removed."
        ),
    )
    print(f"\nDone: {commit.commit_url if hasattr(commit, 'commit_url') else commit}")
    print("\nThe Space will now rebuild (a few minutes — the image builds the dataset")
    print("and trains the models). Watch the Logs tab.")
    print("\nAfter it builds, check Settings:")
    print("  - MAPPLS_API_KEY must still be set as a Secret, or the map will not render.")
    print("  - ENABLE_SIMULATOR is not set, so the demo starts with an empty learning")
    print("    log and no CCTV alert. Set it to 1 as a Variable if you want live motion;")
    print("    simulated rows are tagged so they cannot be mistaken for real feedback.")


if __name__ == "__main__":
    main()
