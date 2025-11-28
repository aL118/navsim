#!/usr/bin/env python3
"""
Analyze PDM score evaluation results from CSV file.

Usage:
    python scripts/evaluation/analyze_results.py <csv_file_path>

Example:
    python scripts/evaluation/analyze_results.py exp/bash/2025.11.27.12.46.20/2025.11.27.12.55.29.csv
"""

import sys
import pandas as pd
from pathlib import Path


def analyze_results(csv_path: str):
    """Analyze PDM score results and print summary statistics."""

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filter to only scenario rows (exclude summary rows)
    scenario_df = df[~df['token'].str.contains('extended_pdm_score', na=False)]

    print("=" * 60)
    print("PDM SCORE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nResults file: {csv_path}")

    # Count total scenarios
    total = len(scenario_df)
    print(f"\nTotal scenarios evaluated: {total}")

    # Count valid scenarios (no crashes)
    valid = scenario_df['valid'].sum()
    failed = total - valid
    print(f"Valid scenarios (no crashes): {valid}")
    print(f"Failed scenarios (crashed): {failed}")

    # Count scenarios with score > 0
    scored = (scenario_df['score'] > 0).sum()
    zero_score = (scenario_df['score'] == 0).sum()
    print(f"\nScenarios with score > 0: {scored}")
    print(f"Scenarios with score = 0: {zero_score}")

    # Average score
    avg_score = scenario_df['score'].mean()
    print(f"\nAverage score (all scenarios): {avg_score:.4f}")

    # Average score (excluding zeros)
    if scored > 0:
        avg_score_nonzero = scenario_df[scenario_df['score'] > 0]['score'].mean()
        print(f"Average score (non-zero only): {avg_score_nonzero:.4f}")

    # Check which multiplicative metrics caused failures
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS (for scenarios with score = 0)")
    print("=" * 60)

    zero_scenarios = scenario_df[scenario_df['score'] == 0]
    print(f"\nTotal scenarios with score = 0: {len(zero_scenarios)}")

    if len(zero_scenarios) > 0:
        # Count failures by metric
        print("\nMultiplicative metric failures:")
        print(f"  No at-fault collisions: {(zero_scenarios['no_at_fault_collisions_stage_one'] == 0).sum()} failures")
        print(f"  Drivable area compliance: {(zero_scenarios['drivable_area_compliance_stage_one'] == 0).sum()} failures")
        print(f"  Driving direction compliance: {(zero_scenarios['driving_direction_compliance_stage_one'] < 1).sum()} failures")
        print(f"  Traffic light compliance: {(zero_scenarios['traffic_light_compliance_stage_one'] == 0).sum()} failures")
        print(f"  Time to collision: {(zero_scenarios['time_to_collision_within_bound_stage_one'] == 0).sum()} failures")

    # Score distribution
    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTION")
    print("=" * 60)
    print(f"\nMin score: {scenario_df['score'].min():.4f}")
    print(f"Max score: {scenario_df['score'].max():.4f}")
    print(f"Median score: {scenario_df['score'].median():.4f}")
    print(f"25th percentile: {scenario_df['score'].quantile(0.25):.4f}")
    print(f"75th percentile: {scenario_df['score'].quantile(0.75):.4f}")

    # Check if two-stage evaluation was performed
    summary_row = df[df['token'] == 'extended_pdm_score_combined']
    if len(summary_row) > 0 and summary_row.iloc[0]['valid']:
        print("\n" + "=" * 60)
        print("EXTENDED PDM SCORE (TWO-STAGE)")
        print("=" * 60)
        combined_score = summary_row.iloc[0]['score']
        print(f"\nFinal extended PDM score: {combined_score:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print(__doc__)
    #     sys.exit(1)

    csv_path = '/fs/nexus-projects/sim2real/aliu/navsim/exp/bash/2025.11.27.18.49.48/2025.11.27.19.00.00.csv' # sys.argv[1]
    analyze_results(csv_path)


"""
transfuser
============================================================
PDM SCORE EVALUATION SUMMARY
============================================================

Results file: /fs/nexus-projects/sim2real/aliu/navsim/exp/bash/2025.11.27.12.15.09/2025.11.27.12.26.11.csv

Total scenarios evaluated: 100
Valid scenarios (no crashes): 100
Failed scenarios (crashed): 0

Scenarios with score > 0: 86
Scenarios with score = 0: 14

Average score (all scenarios): 0.7915
Average score (non-zero only): 0.9203

============================================================
FAILURE ANALYSIS (for scenarios with score = 0)
============================================================

Total scenarios with score = 0: 14

Multiplicative metric failures:
  No at-fault collisions: 0 failures
  Drivable area compliance: 2 failures
  Driving direction compliance: 1 failures
  Traffic light compliance: 0 failures
  Time to collision: 1 failures

============================================================
SCORE DISTRIBUTION
============================================================

Min score: 0.0000
Max score: 1.0000
Median score: 0.9182
25th percentile: 0.8425
75th percentile: 0.9845

============================================================


carla_garage
============================================================
PDM SCORE EVALUATION SUMMARY
============================================================

Results file: /fs/nexus-projects/sim2real/aliu/navsim/exp/bash/2025.11.27.12.46.20/2025.11.27.12.55.29.csv

Total scenarios evaluated: 100
Valid scenarios (no crashes): 100
Failed scenarios (crashed): 0

Scenarios with score > 0: 79
Scenarios with score = 0: 21

Average score (all scenarios): 0.6472
Average score (non-zero only): 0.8193

============================================================
FAILURE ANALYSIS (for scenarios with score = 0)
============================================================

Total scenarios with score = 0: 21

Multiplicative metric failures:
  No at-fault collisions: 8 failures
  Drivable area compliance: 11 failures
  Driving direction compliance: 1 failures
  Traffic light compliance: 2 failures
  Time to collision: 8 failures

============================================================
SCORE DISTRIBUTION
============================================================

Min score: 0.0000
Max score: 1.0000
Median score: 0.8037
25th percentile: 0.5568
75th percentile: 0.8619

============================================================


sim2drive_mixdata
============================================================
PDM SCORE EVALUATION SUMMARY
============================================================

Results file: /fs/nexus-projects/sim2real/aliu/navsim/exp/bash/sim2drive_mixdata/2025.11.27.15.11.43.csv

Total scenarios evaluated: 100
Valid scenarios (no crashes): 100
Failed scenarios (crashed): 0

Scenarios with score > 0: 64
Scenarios with score = 0: 36

Average score (all scenarios): 0.5280
Average score (non-zero only): 0.8249

============================================================
FAILURE ANALYSIS (for scenarios with score = 0)
============================================================

Total scenarios with score = 0: 36

Multiplicative metric failures:
  No at-fault collisions: 7 failures
  Drivable area compliance: 26 failures
  Driving direction compliance: 3 failures
  Traffic light compliance: 1 failures
  Time to collision: 9 failures

============================================================
SCORE DISTRIBUTION
============================================================

Min score: 0.0000
Max score: 1.0000
Median score: 0.7864
25th percentile: 0.0000
75th percentile: 0.8571

============================================================

sim2drive
============================================================
PDM SCORE EVALUATION SUMMARY
============================================================

Results file: /fs/nexus-projects/sim2real/aliu/navsim/exp/bash/sim2drive/2025.11.27.16.20.51.csv

Total scenarios evaluated: 100
Valid scenarios (no crashes): 100
Failed scenarios (crashed): 0

Scenarios with score > 0: 76
Scenarios with score = 0: 24

Average score (all scenarios): 0.5936
Average score (non-zero only): 0.7810

============================================================
FAILURE ANALYSIS (for scenarios with score = 0)
============================================================

Total scenarios with score = 0: 24

Multiplicative metric failures:
  No at-fault collisions: 8 failures
  Drivable area compliance: 17 failures
  Driving direction compliance: 0 failures
  Traffic light compliance: 0 failures
  Time to collision: 9 failures

============================================================
SCORE DISTRIBUTION
============================================================

Min score: 0.0000
Max score: 1.0000
Median score: 0.7702
25th percentile: 0.2568
75th percentile: 0.8433

============================================================
"""