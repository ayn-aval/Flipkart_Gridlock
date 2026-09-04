"""
CCTV Vision Worker
==================
Runs YOLOv8 inference in a separate process, one frame at a time.

Why this exists. The API process imports xgboost and scikit-learn at startup; the
vision engine imports torch. Each ships its own OpenMP runtime, and on macOS loading
one after the other and then entering a parallel region from a request worker thread
aborts the process outright — the CCTV endpoints hang and then take the whole server
down with no traceback.

Isolating inference in a child process fixes that and is better engineering anyway:

  * torch never loads in the web process, so boot stays fast and memory lower
  * a native crash in the CV stack kills one worker, not the API
  * inference is naturally serialised, so a burst of camera clicks cannot thrash CPU

This module must NOT import torch, cv2 or ultralytics at module level — it is imported
by the parent process, and the whole point is to keep those out of it. The import
happens inside `run_analysis`, which only ever executes in the child.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

# One worker: inference is CPU-bound and the model is held in the child's memory.
_pool: Optional[ProcessPoolExecutor] = None

# Generous enough for a cold child (spawn + torch import + weight load), short enough
# that a wedged worker surfaces as an error instead of hanging the request forever.
ANALYSIS_TIMEOUT_S = 120


def run_analysis(image_bytes: bytes) -> dict:
    """Executed in the CHILD process. Imports the heavy stack there, not in the parent."""
    from backend.cv_engine import analyze_cctv_frame
    return analyze_cctv_frame(image_bytes)


def warm_up() -> bool:
    """Executed in the CHILD process: load the weights so the first request is fast."""
    from backend.cv_engine import get_yolo_model
    get_yolo_model()
    return True


def get_pool() -> ProcessPoolExecutor:
    """One worker, started with 'spawn'.

    The start method must be explicit. Linux defaults to 'fork', which would hand the
    child a copy of the parent's address space — including xgboost's already-initialised
    OpenMP runtime — and forking from a thread with an active OpenMP pool is the exact
    condition that deadlocks. macOS already defaults to 'spawn', so relying on the
    default would have made this work in local testing and hang in the Linux container.
    'spawn' gives the child a clean interpreter that imports torch first and nothing else.
    """
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def analyze(image_bytes: bytes) -> dict:
    """Analyse a frame in the worker process.

    Raises RuntimeError with a readable message if the worker dies or times out, so
    the endpoint can return a 500 instead of the server disappearing.
    """
    try:
        return get_pool().submit(run_analysis, image_bytes).result(timeout=ANALYSIS_TIMEOUT_S)
    except Exception as exc:
        # A crashed worker leaves the pool permanently broken; drop it so the next
        # request gets a fresh process rather than a persistent failure.
        global _pool
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None
        raise RuntimeError(f"Vision worker failed: {exc}") from exc


def warm() -> bool:
    """Kick the worker into existence and preload the model. Safe to call at startup."""
    try:
        return bool(get_pool().submit(warm_up).result(timeout=ANALYSIS_TIMEOUT_S))
    except Exception:
        global _pool
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None
        return False


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
