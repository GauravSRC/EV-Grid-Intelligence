"""EV Grid Intelligence - end-to-end pipeline CLI.

Usage
-----
    python run_all.py --quick            # <= ~3 min on cached data (reduced weeks / MC paths)
    python run_all.py --full             # full-year runs, all scenarios (<= ~20 min)
    python run_all.py --stage simulate   # run one stage (data|forecast|calibrate|simulate|analysis|report)
    python run_all.py --full --force     # ignore stage caches

Finished stages are cached in results/cache/ (git-ignored) so reruns resume.
"""
from __future__ import annotations

import argparse
import logging
import time

from evgrid.config import CACHE_DIR, setup_logging

FORCE_DATA = False
STAGES = ["data", "forecast", "calibrate", "simulate", "analysis", "report"]
log = logging.getLogger("run_all")


def run_stage(stage: str, mode: str, force: bool) -> None:
    marker = CACHE_DIR / f"stage_{stage}_{mode}.done"
    if marker.exists() and not force:
        log.info("stage %s (%s) cached - skipping (use --force to rerun)", stage, mode)
        return
    t0 = time.time()
    if stage == "data":
        from evgrid.pipeline import stage_data
        stage_data(force=force and FORCE_DATA)
    elif stage == "forecast":
        from evgrid.pipeline import stage_forecast
        stage_forecast(mode=mode, force=force)
    elif stage == "calibrate":
        from evgrid.pipeline import stage_calibrate
        stage_calibrate(mode=mode, force=force)
    elif stage == "simulate":
        from evgrid.pipeline import stage_simulate
        stage_simulate(mode=mode)
    elif stage == "analysis":
        from evgrid.pipeline import stage_analysis
        stage_analysis(mode=mode)
    elif stage == "report":
        from evgrid.reporting.build_report import build_report
        build_report()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{time.time() - t0:.1f}s", encoding="utf-8")
    log.info("stage %s done in %.1f s", stage, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="fast mode (default)")
    g.add_argument("--full", action="store_true", help="full mode")
    ap.add_argument("--stage", choices=STAGES, help="run a single stage")
    ap.add_argument("--force", action="store_true", help="ignore caches")
    args = ap.parse_args()
    global FORCE_DATA
    FORCE_DATA = args.force
    setup_logging()
    mode = "full" if args.full else "quick"
    stages = [args.stage] if args.stage else STAGES
    t0 = time.time()
    for s in stages:
        run_stage(s, mode, args.force or bool(args.stage))
    log.info("pipeline finished in %.1f s (mode=%s)", time.time() - t0, mode)


if __name__ == "__main__":
    main()
