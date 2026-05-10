"""Download Jena Climate, resample to hourly, save to data/processed/."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import prepare_jena  # noqa: E402


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    out = prepare_jena(raw_dir, processed_dir)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
