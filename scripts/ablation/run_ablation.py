"""Run the ablation study and write per-variant JSON + a comparison table.

Each variant's full metric block is written to results/ablation/<name>.json and
a combined results/ablation/summary.json holds every variant plus a flattened
comparison table (grouped by ablation axis). Uses a reduced, shared epoch budget
so all variants run in one sitting; pass --epochs to change it, --smoke for a
1-epoch sanity pass.

  python scripts/ablation/run_ablation.py
  python scripts/ablation/run_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ablation import comparison_table, default_variants, run_variant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--variants", type=str, default=None,
        help="comma-separated variant names to (re)run; default: all. summary.json "
             "is always rebuilt from every per-variant file on disk, so a targeted "
             "run preserves the rows it did not recompute.",
    )
    args = parser.parse_args()

    epochs = 1 if args.smoke else args.epochs
    all_specs = default_variants(epochs=epochs)
    known_names = [s.name for s in all_specs]

    if args.variants:
        wanted = {n.strip() for n in args.variants.split(",") if n.strip()}
        unknown = wanted - set(known_names)
        if unknown:
            raise SystemExit(f"Unknown variant(s): {sorted(unknown)}. Known: {known_names}")
        specs = [s for s in all_specs if s.name in wanted]
    else:
        specs = all_specs

    out_dir = ROOT / "results"
    var_dir = out_dir / "ablation"
    var_dir.mkdir(parents=True, exist_ok=True)

    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.axis:11s} {spec.name} ...")
        res = run_variant(spec)
        (var_dir / f"{spec.name}.json").write_text(json.dumps(res, indent=2))
        print(f"    crps={res['crps']:.3f} pinball={res['pinball']:.3f} "
              f"picp={res['picp']:.3f} mis={res['mis']:.3f} rmse={res['rmse']:.3f}")

    # Rebuild summary.json from all per-variant files on disk (in the known
    # variant order) so a targeted --variants run keeps the untouched rows.
    all_results = []
    for name in known_names:
        p = var_dir / f"{name}.json"
        if p.exists():
            all_results.append(json.loads(p.read_text()))
    table = comparison_table(all_results)
    summary = {"epochs": epochs, "smoke": bool(args.smoke), "variants": all_results, "table": table}
    (var_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nComparison table (by axis):")
    cur_axis = None
    for row in table:
        if row["axis"] != cur_axis:
            cur_axis = row["axis"]
            print(f"\n[{cur_axis}]")
        print(f"  {row['name']:26s} crps={row['crps']:.3f} pinball={row['pinball']:.3f} "
              f"picp={row['picp']:.3f} mis={row['mis']:.3f} rmse={row['rmse']:.3f}")
    print(f"\nSaved -> {var_dir / 'summary.json'} (+ per-variant files in {var_dir})")


if __name__ == "__main__":
    main()
