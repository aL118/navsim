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
import numpy as np
from pathlib import Path


def analyze_results(csv_path: str):
    """Analyze PDM score results and print summary statistics."""

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filter to only scenario rows (exclude summary rows)
    scenario_df = df[~df['token'].str.contains('extended_pdm_score', na=False)]

    # Separate stage-one and stage-two scenarios
    stage_one_df = scenario_df[scenario_df['no_at_fault_collisions_stage_one'].notna()]
    stage_two_df = scenario_df[scenario_df['no_at_fault_collisions_stage_two'].notna()]

    print("=" * 80)
    print("PDM SCORE RESULTS")
    print("=" * 80)
    print(f"File: {csv_path}\n")

    # EPDMS Scores (most important)
    print("EXTENDED PDM SCORES (EPDMS):")

    # Check for official EPDMS scores
    epdms_combined_row = df[df['token'] == 'extended_pdm_score_combined']

    if len(epdms_combined_row) > 0 and epdms_combined_row.iloc[0]['valid']:
        # Official EPDMS available
        epdms_stage_one_row = df[df['token'] == 'extended_pdm_score_stage_one']
        epdms_stage_two_row = df[df['token'] == 'extended_pdm_score_stage_two']

        if len(epdms_stage_one_row) > 0 and not pd.isna(epdms_stage_one_row.iloc[0]['score']):
            print(f"  Stage-One:  {epdms_stage_one_row.iloc[0]['score']:.4f}")
        if len(epdms_stage_two_row) > 0 and not pd.isna(epdms_stage_two_row.iloc[0]['score']):
            print(f"  Stage-Two:  {epdms_stage_two_row.iloc[0]['score']:.4f}")
        print(f"  Combined:   {epdms_combined_row.iloc[0]['score']:.4f}")
    else:
        # Estimate EPDMS
        print("  (Estimated - true EPDMS requires reactive mapping weights)")

        if len(stage_one_df) > 0:
            stage_one_scores = stage_one_df[stage_one_df['valid'] == True]['score'].dropna()
            if len(stage_one_scores) > 0:
                print(f"  Stage-One:  {stage_one_scores.mean():.4f} (n={len(stage_one_scores)})")

        if len(stage_two_df) > 0:
            stage_two_scores = stage_two_df[stage_two_df['valid'] == True]['score'].dropna()
            if len(stage_two_scores) > 0:
                print(f"  Stage-Two:  {stage_two_scores.mean():.4f} (n={len(stage_two_scores)})")

        all_scores = scenario_df[scenario_df['valid'] == True]['score'].dropna()
        if len(all_scores) > 0:
            print(f"  Combined:   {all_scores.mean():.4f} (n={len(all_scores)})")

    # Average scores
    print(f"\nAVERAGE SCORE: {scenario_df['score'].mean():.4f}")
    valid_scenarios = scenario_df[scenario_df['valid'] == True]
    if len(valid_scenarios) > 0:
        print(f"  (Valid scenarios only): {valid_scenarios['score'].mean():.4f}")

    # Count infractions (scenarios with zero score)
    print(f"\nINFRACTIONS (Zero Score Scenarios): {(scenario_df['score'] == 0).sum()}/{len(scenario_df)}")

    # Show which metrics caused failures
    zero_scenarios = scenario_df[scenario_df['score'] == 0]
    if len(zero_scenarios) > 0:
        zero_stage_two = zero_scenarios[zero_scenarios['no_at_fault_collisions_stage_two'].notna()]

        if len(zero_stage_two) > 0:
            print(f"  Main failure causes (stage-two):")
            collisions = (zero_stage_two['no_at_fault_collisions_stage_two'] == 0).sum()
            drivable = (zero_stage_two['drivable_area_compliance_stage_two'] == 0).sum()
            ttc = (zero_stage_two['time_to_collision_within_bound_stage_two'] == 0).sum()

            if collisions > 0:
                print(f"    Collisions: {collisions}")
            if drivable > 0:
                print(f"    Off-road: {drivable}")
            if ttc > 0:
                print(f"    TTC violations: {ttc}")

    # Summary stats
    total = len(scenario_df)
    valid = scenario_df['valid'].sum()
    print(f"\nSUMMARY:")
    print(f"  Total scenarios: {total} (stage-one: {len(stage_one_df)}, stage-two: {len(stage_two_df)})")
    print(f"  Valid (no crash): {valid}/{total}")
    print(f"  Failed (crashed): {total - valid}/{total}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print(__doc__)
    #     sys.exit(1)

    csv_path = '/fs/nexus-projects/sim2real/aliu/navsim/exp/mixdata_sim2drive/2025.11.30.00.07.24/2025.11.30.00.17.05.csv' # sys.argv[1]
    analyze_results(csv_path)

"""
================================================================================
FULL PDM SCORE RESULTS
================================================================================
File: /fs/nexus-projects/sim2real/aliu/navsim/exp/full_no_aug/2025.11.29.19.03.14/2025.11.29.19.14.21.csv

EXTENDED PDM SCORES (EPDMS):
  (Estimated - true EPDMS requires reactive mapping weights)
  Stage-Two:  0.2194 (n=103)
  Combined:   0.2194 (n=103)

AVERAGE SCORE: 0.2194
  (Valid scenarios only): 0.2194

INFRACTIONS (Zero Score Scenarios): 71/111
  Main failure causes (stage-two):
    Collisions: 13
    Off-road: 62
    TTC violations: 16

SUMMARY:
  Total scenarios: 111 (stage-one: 0, stage-two: 103)
  Valid (no crash): 103/111
  Failed (crashed): 8/111

================================================================================

================================================================================
SIM2DRIVE PDM SCORE RESULTS
================================================================================
File: /fs/nexus-projects/sim2real/aliu/navsim/exp/sim2drive_virtual_only/2025.11.29.19.14.36/2025.11.29.19.24.18.csv

EXTENDED PDM SCORES (EPDMS):
  (Estimated - true EPDMS requires reactive mapping weights)
  Stage-Two:  0.4883 (n=83)
  Combined:   0.4883 (n=83)

AVERAGE SCORE: 0.4883
  (Valid scenarios only): 0.4883

INFRACTIONS (Zero Score Scenarios): 27/91
  Main failure causes (stage-two):
    Collisions: 5
    Off-road: 20
    TTC violations: 5

SUMMARY:
  Total scenarios: 91 (stage-one: 0, stage-two: 83)
  Valid (no crash): 83/91
  Failed (crashed): 8/91

================================================================================

================================================================================
MIXDATA PDM SCORE RESULTS
================================================================================
File: /fs/nexus-projects/sim2real/aliu/navsim/exp/sim2drive_virtual_only/2025.11.29.23.09.59/2025.11.29.23.19.45.csv

EXTENDED PDM SCORES (EPDMS):
  (Estimated - true EPDMS requires reactive mapping weights)
  Stage-Two:  0.3392 (n=96)
  Combined:   0.3392 (n=96)

AVERAGE SCORE: 0.3392
  (Valid scenarios only): 0.3392

INFRACTIONS (Zero Score Scenarios): 48/103
  Main failure causes (stage-two):
    Collisions: 4
    Off-road: 42
    TTC violations: 4

SUMMARY:
  Total scenarios: 103 (stage-one: 0, stage-two: 96)
  Valid (no crash): 96/103
  Failed (crashed): 7/103

================================================================================
BASELINE PDM SCORE RESULTS
================================================================================
File: /fs/nexus-projects/sim2real/aliu/navsim/exp/full_no_aug/2025.11.29.22.55.34/2025.11.29.23.09.43.csv

EXTENDED PDM SCORES (EPDMS):
  (Estimated - true EPDMS requires reactive mapping weights)
  Stage-Two:  0.2568 (n=120)
  Combined:   0.2568 (n=120)

AVERAGE SCORE: 0.2568
  (Valid scenarios only): 0.2568

INFRACTIONS (Zero Score Scenarios): 79/131
  Main failure causes (stage-two):
    Collisions: 27
    Off-road: 57
    TTC violations: 32

SUMMARY:
  Total scenarios: 131 (stage-one: 0, stage-two: 120)
  Valid (no crash): 120/131
  Failed (crashed): 11/131

================================================================================

================================================================================
MIX_SIMDRIVE SCORE RESULTS
================================================================================
File: /fs/nexus-projects/sim2real/aliu/navsim/exp/mixdata_sim2drive/2025.11.30.00.07.24/2025.11.30.00.17.05.csv

EXTENDED PDM SCORES (EPDMS):
  (Estimated - true EPDMS requires reactive mapping weights)
  Stage-Two:  0.3840 (n=61)
  Combined:   0.3840 (n=61)

AVERAGE SCORE: 0.3840
  (Valid scenarios only): 0.3840

INFRACTIONS (Zero Score Scenarios): 32/68
  Main failure causes (stage-two):
    Collisions: 17
    Off-road: 15
    TTC violations: 21

SUMMARY:
  Total scenarios: 68 (stage-one: 0, stage-two: 61)
  Valid (no crash): 61/68
  Failed (crashed): 7/68

================================================================================
"""