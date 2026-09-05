"""Config loading: rubric, thresholds, and agent settings live in YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DB_PATH = DATA_DIR / "second_reader.duckdb"

MAX_ITEMS = 150
N_ANNOTATORS = 15


@dataclass(frozen=True)
class Rubric:
    version: str
    dimensions: dict[str, dict]  # name -> {weight, description}
    verdict_rule: str

    @property
    def dimension_names(self) -> list[str]:
        return list(self.dimensions.keys())


@dataclass(frozen=True)
class Thresholds:
    high: float
    low: float
    flip_penalty: float
    max_tool_calls: int
    truncate_chars: int
    history_limit: int
    verifier_max_tool_calls: int
    verifier_overturn_penalty: float
    sweep_high: list[float]
    sweep_low: list[float]
    model: str
    pricing: dict[str, dict[str, float]]


def load_rubric(path: Path | None = None) -> Rubric:
    path = path or (CONFIG_DIR / "rubric_v1.yaml")
    raw = yaml.safe_load(path.read_text())
    return Rubric(
        version=raw["version"],
        dimensions=raw["dimensions"],
        verdict_rule=raw["verdict_rule"],
    )


def load_thresholds(path: Path | None = None) -> Thresholds:
    path = path or (CONFIG_DIR / "thresholds.yaml")
    raw = yaml.safe_load(path.read_text())
    r, a, v = raw["routing"], raw["agent"], raw["verifier"]
    return Thresholds(
        high=float(r["high"]),
        low=float(r["low"]),
        flip_penalty=float(r["flip_penalty"]),
        max_tool_calls=int(a["max_tool_calls"]),
        truncate_chars=int(a["truncate_chars"]),
        history_limit=int(a["history_limit"]),
        verifier_max_tool_calls=int(v["max_tool_calls"]),
        verifier_overturn_penalty=float(v["overturn_penalty"]),
        sweep_high=[float(x) for x in raw["sweep"]["high"]],
        sweep_low=[float(x) for x in raw["sweep"]["low"]],
        model=raw["model"],
        pricing=raw["pricing"],
    )


def load_api_key() -> str | None:
    """ANTHROPIC_API_KEY from env, else from a project-local .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
