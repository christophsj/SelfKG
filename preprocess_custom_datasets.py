#!/usr/bin/env python
# coding: UTF-8
"""
Preprocessing script for custom datasets in data/custom/
This script processes raw entity data and generates LaBSE embeddings and neighbor information.
"""

import os
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from os.path import join
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from settings import DATA_DIR, LaBSE_DIM, NEIGHBOR_SIZE

MAX_LEN = 130


class LaBSEEncoder:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            join(DATA_DIR, "LaBSE"), do_lower_case=False
        )
        self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)
        self.model.eval()

    def encode(self, sentences):
        with torch.no_grad():
            tok_res = self.tokenizer(
                sentences,
                add_special_tokens=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
            input_ids = torch.LongTensor(
                [d[:MAX_LEN] for d in tok_res["input_ids"]]
            ).to(self.device)
            token_type_ids = torch.LongTensor(tok_res["token_type_ids"]).to(self.device)
            attention_mask = torch.LongTensor(tok_res["attention_mask"]).to(self.device)
            output = self.model(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            return F.normalize(output[0][:, 1:-1, :].sum(dim=1))


def clean_entity_ids(dataset_path, doc_id):
    """
    Create cleaned_ent_ids files from ent_ids files.
    Format: id \t entity_name
    """
    ent_ids_file = join(dataset_path, f"ent_ids_{doc_id}")
    cleaned_file = join(dataset_path, f"cleaned_ent_ids_{doc_id}")

    print(f"Creating {cleaned_file}...")

    # Read entity IDs and extract entity names from URIs
    with open(ent_ids_file, "r", encoding="utf-8") as f_in, open(
        cleaned_file, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_id = parts[0]
                uri = parts[1]
                # Extract entity name from URI (last part after /)
                entity_name = uri.split("/")[-1].replace("_", " ")
                f_out.write(f"{entity_id}\t{entity_name}\n")


def load_entity_names(dataset_path, doc_id):
    """Load entity ID to name mapping."""
    cleaned_file = join(dataset_path, f"cleaned_ent_ids_{doc_id}")
    id_entity = {}

    with open(cleaned_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_id = int(parts[0])
                entity_name = parts[1]
                id_entity[entity_id] = entity_name

    return id_entity


def create_ref_files(dataset_path):
    """
    Create test.ref and valid.ref files from ref_pairs and sup_pairs.
    """
    ref_pairs_file = join(dataset_path, "ref_pairs")
    sup_pairs_file = join(dataset_path, "sup_pairs")
    test_ref_file = join(dataset_path, "test.ref")
    valid_ref_file = join(dataset_path, "valid.ref")

    print(f"Creating reference files...")

    # Use ref_pairs for test set
    if os.path.exists(ref_pairs_file):
        with open(ref_pairs_file, "r", encoding="utf-8") as f_in, open(
            test_ref_file, "w", encoding="utf-8"
        ) as f_out:
            f_out.write(f_in.read())

    # Use sup_pairs for validation set (or a subset of ref_pairs if sup_pairs doesn't exist)
    if os.path.exists(sup_pairs_file):
        with open(sup_pairs_file, "r", encoding="utf-8") as f_in, open(
            valid_ref_file, "w", encoding="utf-8"
        ) as f_out:
            f_out.write(f_in.read())
    else:
        # If no sup_pairs, use first 20% of ref_pairs for validation
        with open(ref_pairs_file, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
        split_point = max(1, len(lines) // 5)
        with open(valid_ref_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(lines[:split_point])


def generate_embeddings(dataset_path, doc_id, encoder):
    """Generate LaBSE embeddings for entities."""
    id_entity = load_entity_names(dataset_path, doc_id)
    output_file = join(dataset_path, f"raw_LaBSE_emb_{doc_id}.pkl")

    print(f"Generating embeddings for document {doc_id}...")

    id_embedding = {}
    batch_size = 32
    entity_ids = list(id_entity.keys())

    for i in tqdm(range(0, len(entity_ids), batch_size)):
        batch_ids = entity_ids[i : i + batch_size]
        batch_names = [id_entity[eid] for eid in batch_ids]

        embeddings = encoder.encode(batch_names).cpu().detach().numpy()

        for j, entity_id in enumerate(batch_ids):
            # Wrap embedding in list to match expected format: id -> [embedding]
            id_embedding[int(entity_id)] = [embeddings[j].tolist()]

    with open(output_file, "wb") as f:
        pickle.dump(id_embedding, f)

    print(f"Saved embeddings to {output_file}")


def preprocess_dataset(dataset_name, device="cuda:0"):
    """
    Preprocess a single dataset.

    Args:
        dataset_name: Name of the dataset folder in data/custom/
        device: Device to use for encoding (cuda:0, cuda:1, cpu, etc.)
    """
    dataset_path = join(DATA_DIR, "custom", dataset_name)

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return False

    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # Step 1: Create cleaned entity ID files
    print("Step 1: Creating cleaned entity ID files...")
    clean_entity_ids(dataset_path, "1")
    clean_entity_ids(dataset_path, "2")

    # Step 2: Create reference files
    print("\nStep 2: Creating reference files...")
    create_ref_files(dataset_path)

    # Step 3: Generate LaBSE embeddings
    print("\nStep 3: Generating LaBSE embeddings...")
    encoder = LaBSEEncoder(device=device)
    generate_embeddings(dataset_path, "1", encoder)
    generate_embeddings(dataset_path, "2", encoder)

    print(f"\n{'='*60}")
    print(f"Completed preprocessing for {dataset_name}")
    print(f"{'='*60}\n")

    return True


def preprocess_all_custom_datasets(device="cuda:0"):
    """
    Preprocess all datasets in data/custom/

    Args:
        device: Device to use for encoding (cuda:0, cuda:1, cpu, etc.)
    """
    custom_dir = join(DATA_DIR, "custom")

    if not os.path.exists(custom_dir):
        print(f"Custom data directory {custom_dir} does not exist!")
        return

    # Check if LaBSE model exists
    labse_path = join(DATA_DIR, "LaBSE")
    if not os.path.exists(labse_path):
        print(f"ERROR: LaBSE model not found at {labse_path}")
        print("Please download the LaBSE model first using:")
        print("  cd data && bash getdata.sh")
        return

    # Get all dataset folders
    datasets = [d for d in os.listdir(custom_dir) if os.path.isdir(join(custom_dir, d))]

    if not datasets:
        print(f"No datasets found in {custom_dir}")
        return

    print(f"Found {len(datasets)} dataset(s) to process: {datasets}")

    for dataset_name in datasets:
        success = preprocess_dataset(dataset_name, device=device)
        if not success:
            print(f"Failed to process {dataset_name}")

    print("\n" + "=" * 60)
    print("All datasets preprocessed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess custom datasets for SelfKG"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to preprocess (e.g., fr_en). If not specified, processes all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for encoding (default: cuda:0)",
    )

    args = parser.parse_args()

    if args.dataset:
        preprocess_dataset(args.dataset, device=args.device)
    else:
        preprocess_all_custom_datasets(device=args.device)
