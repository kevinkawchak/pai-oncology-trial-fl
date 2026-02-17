#!/usr/bin/env python3
"""Generate synthetic oncology datasets for federated learning.

Produces CSV files with de-identified clinical features, suitable for
testing the federated training pipeline.

Usage:
    python examples/generate_synthetic_data.py
    python examples/generate_synthetic_data.py --output-dir data/ --num-sites 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data

FEATURE_NAMES = [
    "tumor_volume_cm3",
    "growth_rate_days",
    "chemo_sensitivity",
    "radio_sensitivity",
    "age_bracket",
    "pdl1_expression",
    "ki67_index",
    "her2_status",
    "er_status",
    "pr_status",
    "white_blood_cell",
    "hemoglobin",
    "platelet_count",
    "albumin",
    "ldh_level",
    "creatinine",
    "bmi",
    "ecog_score",
    "stage_numeric",
    "grade_numeric",
    "lymph_node_count",
    "margin_status",
    "necrosis_pct",
    "mitotic_rate",
    "vascular_invasion",
    "perineural_invasion",
    "smoking_status",
    "comorbidity_index",
    "prior_treatments",
    "time_since_diagnosis_days",
]

CLASS_NAMES = ["favorable_response", "poor_response"]


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic oncology data for {args.num_sites} sites...")

    # Generate data
    n_samples = args.num_sites * args.samples_per_site
    x, y = generate_synthetic_oncology_data(
        n_samples=n_samples,
        n_features=30,
        n_classes=2,
        seed=args.seed,
    )

    # Partition across sites
    partitioner = DataPartitioner(
        num_sites=args.num_sites,
        strategy=args.strategy,
        test_fraction=0.2,
        seed=args.seed,
    )
    sites = partitioner.partition(x, y)

    # Save each site's data
    for site in sites:
        # Training data
        train_df = pd.DataFrame(site.x_train, columns=FEATURE_NAMES)
        train_df["label"] = site.y_train
        train_df["label_name"] = [CLASS_NAMES[i] for i in site.y_train]
        train_path = output_dir / f"{site.site_id}_train.csv"
        train_df.to_csv(train_path, index=False)

        # Test data
        test_df = pd.DataFrame(site.x_test, columns=FEATURE_NAMES)
        test_df["label"] = site.y_test
        test_df["label_name"] = [CLASS_NAMES[i] for i in site.y_test]
        test_path = output_dir / f"{site.site_id}_test.csv"
        test_df.to_csv(test_path, index=False)

        print(
            f"  {site.site_id}: {site.num_train} train -> {train_path}, "
            f"{site.num_test} test -> {test_path}"
        )

    # Save combined metadata
    meta = {
        "num_sites": args.num_sites,
        "samples_per_site": args.samples_per_site,
        "strategy": args.strategy,
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
    }
    import json

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nData saved to {output_dir}/")
    print(f"Total samples: {n_samples}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic oncology data")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num-sites", type=int, default=3, help="Number of sites")
    parser.add_argument("--samples-per-site", type=int, default=200, help="Samples per site")
    parser.add_argument(
        "--strategy", choices=["iid", "non_iid"], default="iid", help="Split strategy"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
