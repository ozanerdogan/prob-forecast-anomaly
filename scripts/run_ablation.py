"""Run the ablation study and write per-variant JSON + a comparison table.

Each variant's full metric block is written to results/ablation/<name>.json and
a combined results/ablation.json holds every variant plus a flattened
comparison table (grouped by ablation axis). Uses a reduced, shared epoch budget
so all variants run in one sitting; pass --epochs to change it, --smoke for a
1-epoch sanity pass.

  python scripts/run_ablation.py
  python scripts/run_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import comparison_table, default_variants, run_variant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    epochs = 1 if args.smoke else args.epochs
    variants = default_variants(epochs=epochs)

    out_dir = ROOT / "results"
    var_dir = out_dir / "ablation"
    var_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, spec in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] {spec.axis:11s} {spec.name} ...")
        res = run_variant(spec)
        (var_dir / f"{spec.name}.json").write_text(json.dumps(res, indent=2))
        results.append(res)
        print(f"    crps={res['crps']:.3f} pinball={res['pinball']:.3f} "
              f"picp={res['picp']:.3f} mis={res['mis']:.3f} rmse={res['rmse']:.3f}")

    table = comparison_table(results)
    summary = {"epochs": epochs, "smoke": bool(args.smoke), "variants": results, "table": table}
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2))

    print("\nComparison table (by axis):")
    cur_axis = None
    for row in table:
        if row["axis"] != cur_axis:
            cur_axis = row["axis"]
            print(f"\n[{cur_axis}]")
        print(f"  {row['name']:26s} crps={row['crps']:.3f} pinball={row['pinball']:.3f} "
              f"picp={row['picp']:.3f} mis={row['mis']:.3f} rmse={row['rmse']:.3f}")
    print(f"\nSaved -> {out_dir / 'ablation.json'} (+ per-variant files in {var_dir})")


if __name__ == "__main__":
    main()
