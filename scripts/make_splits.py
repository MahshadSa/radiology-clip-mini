from __future__ import annotations

import argparse
from pathlib import Path

from src.data import (
    load_iu_xray_flat,
    build_indices_with_subsample,
    save_splits_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-level train/val/test splits for IU X-Ray with a ~N-example subsample."
    )
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Target number of examples (keeps whole patients until ~this size).")
    parser.add_argument("--val_frac", type=float, default=0.10,
                        help="Validation fraction by patient (0–1).")
    parser.add_argument("--test_frac", type=float, default=0.10,
                        help="Test fraction by patient (0–1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--out", type=str, default="data/splits.json",
                        help="Output JSON path for the splits.")
    parser.add_argument("--preview", action="store_true",
                        help="Print split sizes to the console.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading IU X-Ray (OpenI) via datasets...")
    data = load_iu_xray_flat()
    n_total = len(data["patient_id"])
    print(f"Total usable examples: {n_total}")

    print(f"Building subsample (~{args.max_samples}) with patient-level splits...")
    splits = build_indices_with_subsample(
        patient_ids=data["patient_id"],
        max_samples=args.max_samples,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    out_path = Path(args.out)
    save_splits_json(splits, out_path)
    print(f"Saved splits to: {out_path.resolve()}")

    if args.preview:
        n_train = len(splits["train"])
        n_val = len(splits["val"])
        n_test = len(splits["test"])
        n_sum = n_train + n_val + n_test
        print(f"Split sizes  ->  train: {n_train} | val: {n_val} | test: {n_test} | total: {n_sum}")


if __name__ == "__main__":
    main()
