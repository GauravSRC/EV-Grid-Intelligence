"""Configuration loading (base.yaml + frozen calibrated.yaml + scenario overrides)."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
FIG_DIR = ROOT / "reports" / "figures"

log = logging.getLogger("evgrid")


def deep_merge(base: Dict[str, Any], over: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(calibrated: bool = True, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load base config, then frozen calibration (if present), then overrides."""
    cfg = read_yaml(CONFIG_DIR / "base.yaml")
    cal = CONFIG_DIR / "calibrated.yaml"
    if calibrated and cal.exists():
        cfg = deep_merge(cfg, read_yaml(cal).get("overrides", {}))
    return deep_merge(cfg, overrides)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
