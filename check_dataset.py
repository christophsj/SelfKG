#!/usr/bin/env python
"""
Diagnostic script to check custom dataset setup and identify issues.
"""

import os
import pickle
import pandas as pd
from os.path import join
from settings import DATA_DIR


def check_dataset(dataset_name, data_dir="custom"):
    """Check dataset files for common issues."""

    dataset_path = join(DATA_DIR, data_dir, dataset_name)

    print("=" * 80)
    print(f"Checking dataset: {dataset_name}")
    print("=" * 80)

    # Check if directory exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False

    print(f"✓ Dataset path exists: {dataset_path}\n")

    # Check required files
    required_files = {
        "ent_ids_1": "Entity IDs for KG1",
        "ent_ids_2": "Entity IDs for KG2",
        "triples_1": "Triples for KG1",
        "triples_2": "Triples for KG2",
        "ref_pairs": "Test reference pairs",
        "test.ref": "Test reference file",
        "valid.ref": "Validation reference file",
        "cleaned_ent_ids_1": "Cleaned entity IDs 1",
        "cleaned_ent_ids_2": "Cleaned entity IDs 2",
        "raw_LaBSE_emb_1.pkl": "Embeddings for KG1",
        "raw_LaBSE_emb_2.pkl": "Embeddings for KG2",
    }

    missing_files = []
    for fname, desc in required_files.items():
        fpath = join(dataset_path, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"✓ {fname:25s} ({size:>8.1f} KB) - {desc}")
        else:
            print(f"❌ {fname:25s} - MISSING - {desc}")
            missing_files.append(fname)

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Run: python preprocess_custom_datasets.py --dataset", dataset_name)
        return False

    print("\n" + "=" * 80)
    print("Checking data consistency...")
    print("=" * 80 + "\n")

    # Load entity IDs
    ent_ids_1 = pd.read_csv(join(dataset_path, "ent_ids_1"), sep="\t", header=None)
    ent_ids_2 = pd.read_csv(join(dataset_path, "ent_ids_2"), sep="\t", header=None)

    print(f"Entity IDs KG1: {len(ent_ids_1)} entities")
    print(f"Entity IDs KG2: {len(ent_ids_2)} entities")

    # Check ID ranges
    kg1_min, kg1_max = ent_ids_1[0].min(), ent_ids_1[0].max()
    kg2_min, kg2_max = ent_ids_2[0].min(), ent_ids_2[0].max()

    print(f"KG1 ID range: {kg1_min} to {kg1_max}")
    print(f"KG2 ID range: {kg2_min} to {kg2_max}")

    # Load embeddings
    print("\nLoading embeddings...")
    with open(join(dataset_path, "raw_LaBSE_emb_1.pkl"), "rb") as f:
        emb_1 = pickle.load(f)
    with open(join(dataset_path, "raw_LaBSE_emb_2.pkl"), "rb") as f:
        emb_2 = pickle.load(f)

    print(f"Embeddings KG1: {len(emb_1)} entities")
    print(f"Embeddings KG2: {len(emb_2)} entities")

    # Check embedding format
    sample_id_1 = list(emb_1.keys())[0]
    sample_emb_1 = emb_1[sample_id_1]

    print(f"\nSample embedding (ID {sample_id_1}):")
    print(f"  Type: {type(sample_emb_1)}")

    if isinstance(sample_emb_1, list):
        if len(sample_emb_1) > 0:
            print(f"  Length: {len(sample_emb_1)}")
            if isinstance(sample_emb_1[0], list):
                print(f"  Format: Nested list [embedding] ✓")
                print(f"  Embedding dimension: {len(sample_emb_1[0])}")
            elif isinstance(sample_emb_1[0], (int, float)):
                print(f"  Format: Flat list embedding ❌")
                print(f"  ⚠️  WARNING: Embeddings should be [[...]] not [...]")
                print(f"  Re-run preprocessing to fix this!")
            else:
                print(f"  Format: Unknown - {type(sample_emb_1[0])}")

    # Check if all entity IDs have embeddings
    missing_emb_1 = set(ent_ids_1[0]) - set(emb_1.keys())
    missing_emb_2 = set(ent_ids_2[0]) - set(emb_2.keys())

    if missing_emb_1:
        print(f"\n❌ {len(missing_emb_1)} entities in KG1 missing embeddings")
        print(f"   Example missing IDs: {list(missing_emb_1)[:5]}")
    else:
        print(f"\n✓ All KG1 entities have embeddings")

    if missing_emb_2:
        print(f"❌ {len(missing_emb_2)} entities in KG2 missing embeddings")
        print(f"   Example missing IDs: {list(missing_emb_2)[:5]}")
    else:
        print(f"✓ All KG2 entities have embeddings")

    # Check reference pairs
    print("\n" + "=" * 80)
    print("Checking reference pairs...")
    print("=" * 80 + "\n")

    test_ref = pd.read_csv(join(dataset_path, "test.ref"), sep="\t", header=None)
    valid_ref = pd.read_csv(join(dataset_path, "valid.ref"), sep="\t", header=None)

    print(f"Test pairs: {len(test_ref)}")
    print(f"Valid pairs: {len(valid_ref)}")

    # Check if reference IDs are in the entity lists
    test_kg1_ids = set(test_ref[0])
    test_kg2_ids = set(test_ref[1])
    all_kg1_ids = set(ent_ids_1[0])
    all_kg2_ids = set(ent_ids_2[0])

    missing_test_kg1 = test_kg1_ids - all_kg1_ids
    missing_test_kg2 = test_kg2_ids - all_kg2_ids

    if missing_test_kg1:
        print(f"\n❌ {len(missing_test_kg1)} test KG1 IDs not in ent_ids_1")
        print(f"   Example: {list(missing_test_kg1)[:5]}")
    else:
        print(f"✓ All test KG1 IDs exist in ent_ids_1")

    if missing_test_kg2:
        print(f"❌ {len(missing_test_kg2)} test KG2 IDs not in ent_ids_2")
        print(f"   Example: {list(missing_test_kg2)[:5]}")
    else:
        print(f"✓ All test KG2 IDs exist in ent_ids_2")

    # Check if reference IDs have embeddings
    missing_emb_test_1 = test_kg1_ids - set(emb_1.keys())
    missing_emb_test_2 = test_kg2_ids - set(emb_2.keys())

    if missing_emb_test_1:
        print(f"\n❌ {len(missing_emb_test_1)} test KG1 entities missing embeddings")
        print(f"   Example IDs: {list(missing_emb_test_1)[:5]}")
    else:
        print(f"✓ All test KG1 entities have embeddings")

    if missing_emb_test_2:
        print(f"❌ {len(missing_emb_test_2)} test KG2 entities missing embeddings")
        print(f"   Example IDs: {list(missing_emb_test_2)[:5]}")
    else:
        print(f"✓ All test KG2 entities have embeddings")

    # Sample reference pairs
    print("\nSample test reference pairs:")
    for idx in range(min(5, len(test_ref))):
        kg1_id = test_ref.iloc[idx, 0]
        kg2_id = test_ref.iloc[idx, 1]
        print(f"  {kg1_id:>6d} <-> {kg2_id:>6d}")

    # Check triples
    print("\n" + "=" * 80)
    print("Checking triples...")
    print("=" * 80 + "\n")

    triples_1 = pd.read_csv(join(dataset_path, "triples_1"), sep="\t", header=None)
    triples_2 = pd.read_csv(join(dataset_path, "triples_2"), sep="\t", header=None)

    print(f"Triples KG1: {len(triples_1)}")
    print(f"Triples KG2: {len(triples_2)}")

    # Check if all triple entities have embeddings
    triple_ent_1 = set(triples_1[0]) | set(triples_1[2])
    triple_ent_2 = set(triples_2[0]) | set(triples_2[2])

    missing_triple_emb_1 = triple_ent_1 - set(emb_1.keys())
    missing_triple_emb_2 = triple_ent_2 - set(emb_2.keys())

    if missing_triple_emb_1:
        print(
            f"\n❌ {len(missing_triple_emb_1)} entities in triples_1 missing embeddings"
        )
        print(f"   This will cause errors during training!")
    else:
        print(f"✓ All entities in triples_1 have embeddings")

    if missing_triple_emb_2:
        print(
            f"❌ {len(missing_triple_emb_2)} entities in triples_2 missing embeddings"
        )
        print(f"   This will cause errors during training!")
    else:
        print(f"✓ All entities in triples_2 have embeddings")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80 + "\n")

    issues = []

    if missing_files:
        issues.append("Missing required files")
    if missing_emb_1 or missing_emb_2:
        issues.append("Entities missing embeddings")
    if missing_test_kg1 or missing_test_kg2:
        issues.append("Reference IDs not in entity lists")
    if missing_emb_test_1 or missing_emb_test_2:
        issues.append("Test entities missing embeddings")
    if missing_triple_emb_1 or missing_triple_emb_2:
        issues.append("Triple entities missing embeddings")

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nRecommended action:")
        print(
            f"  python preprocess_custom_datasets.py --dataset {dataset_name} --device cuda:0"
        )
        return False
    else:
        print("✓ Dataset looks good!")
        print("\nYou can proceed with training:")
        print(
            f"  python run_custom_experiments.py --dataset {dataset_name} --num_runs 3 --device cuda:0"
        )
        return True


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Check custom dataset setup")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fr_en",
        help="Dataset name to check (default: fr_en)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="custom",
        help="Data directory (default: custom)",
    )

    args = parser.parse_args()

    check_dataset(args.dataset, args.data_dir)
