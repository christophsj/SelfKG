# coding: UTF-8
"""
Modified version of run_LaBSE_neighbor.py for custom datasets.
This script allows running experiments on datasets in data/custom/
"""

from model.layers_LaBSE_neighbor import Trainer, parse_options
import argparse
from settings import DATA_DIR, PROJ_DIR
from os.path import join
import os


class CustomTrainer(Trainer):
    """Extended Trainer that supports custom datasets."""

    def __init__(self, training=True, seed=37, data_dir="custom"):
        self.data_dir = data_dir
        # Temporarily replace the DBP15K import in the parent class
        super().__init__(training=training, seed=seed)

    def _init_data_loaders(self):
        """Override to use custom data loaders."""
        import torch.utils.data as Data
        from loader.CustomRawNeighbors import CustomRawNeighbors
        from script.preprocess.deal_raw_dataset import MyRawdataset

        # Load dataset 1
        loader1 = CustomRawNeighbors(self.args.language, "1", data_dir=self.data_dir)
        myset1 = MyRawdataset(loader1.id_neighbors_dict, loader1.id_adj_tensor_dict)
        del loader1

        self.loader1 = Data.DataLoader(
            dataset=myset1,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        del myset1

        # Load dataset 2
        loader2 = CustomRawNeighbors(self.args.language, "2", data_dir=self.data_dir)
        myset2 = MyRawdataset(loader2.id_neighbors_dict, loader2.id_adj_tensor_dict)
        del loader2

        self.loader2 = Data.DataLoader(
            dataset=myset2,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        del myset2


def parse_custom_options():
    """Parse command line options for custom datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--language", type=str, default="fr_en", help="Dataset name in data/custom/"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="custom",
        help="Data directory (custom or DBP15K)",
    )
    parser.add_argument("--model_language", type=str, default=None)
    parser.add_argument("--model", type=str, default="LaBSE")

    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--queue_length", type=int, default=64)

    parser.add_argument("--center_norm", type=bool, default=False)
    parser.add_argument("--neighbor_norm", type=bool, default=True)
    parser.add_argument("--emb_norm", type=bool, default=True)
    parser.add_argument("--combine", type=bool, default=True)

    parser.add_argument("--gat_num", type=int, default=1)

    parser.add_argument("--t", type=float, default=0.08)
    parser.add_argument("--momentum", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument(
        "--log_dir", type=str, default=None, help="Custom log directory"
    )
    parser.add_argument(
        "--run_id", type=int, default=1, help="Run ID for multiple repetitions"
    )
    parser.add_argument("--time", type=str, default=None)

    args = parser.parse_args()

    # Set model_language to language if not specified
    if args.model_language is None:
        args.model_language = args.language

    # Set time if not specified
    if args.time is None:
        from datetime import datetime

        args.time = datetime.now().strftime("%Y%m%d%H%M%S")

    return args


if __name__ == "__main__":
    import sys
    from datetime import datetime
    import logging
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from settings import fix_seed, VOCAB_SIZE, LaBSE_DIM
    from loader.CustomRawNeighbors import CustomRawNeighbors
    from script.preprocess.deal_raw_dataset import MyRawdataset
    import torch.utils.data as Data
    import pandas as pd
    from tqdm import tqdm
    import random
    import faiss
    import numpy as np

    # Parse arguments
    args = parse_custom_options()

    # Set random seed
    fix_seed(37)

    # Set up device
    device = torch.device(args.device)

    # Check if dataset exists and is preprocessed
    dataset_path = join(DATA_DIR, args.data_dir, args.language)
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path {dataset_path} does not exist!")
        sys.exit(1)

    required_files = [
        "raw_LaBSE_emb_1.pkl",
        "raw_LaBSE_emb_2.pkl",
        "test.ref",
        "valid.ref",
    ]
    for fname in required_files:
        if not os.path.exists(join(dataset_path, fname)):
            print(f"ERROR: Required file {fname} not found in {dataset_path}")
            print(f"Please run preprocessing first:")
            print(f"  python preprocess_custom_datasets.py --dataset {args.language}")
            sys.exit(1)

    # Set up logging
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = join(PROJ_DIR, "log", "custom_experiments", args.language)

    os.makedirs(log_dir, exist_ok=True)

    log_filename = join(log_dir, f"run_{args.run_id}_{args.time}.log")

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    logging.info(f"Starting experiment for dataset: {args.language}")
    logging.info(f"Run ID: {args.run_id}")
    logging.info(f"Arguments: {args}")

    # Create trainer (using original Trainer class with modifications)
    from model.layers_LaBSE_neighbor import (
        MyEmbedder,
        NCESoftmaxLoss,
        BatchMultiHeadGraphAttention,
    )

    # Load data
    logging.info("Loading dataset 1...")
    loader1 = CustomRawNeighbors(args.language, "1", data_dir=args.data_dir)
    myset1 = MyRawdataset(loader1.id_neighbors_dict, loader1.id_adj_tensor_dict)
    del loader1

    data_loader1 = Data.DataLoader(
        dataset=myset1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    eval_loader1 = Data.DataLoader(
        dataset=myset1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    del myset1

    logging.info("Loading dataset 2...")
    loader2 = CustomRawNeighbors(args.language, "2", data_dir=args.data_dir)
    myset2 = MyRawdataset(loader2.id_neighbors_dict, loader2.id_adj_tensor_dict)
    del loader2

    data_loader2 = Data.DataLoader(
        dataset=myset2,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    eval_loader2 = Data.DataLoader(
        dataset=myset2,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    del myset2

    # Load link data
    def link_loader(mode, data_dir, valid=False):
        link = {}
        if valid == False:
            f = "test.ref"
        else:
            f = "valid.ref"
        link_data = pd.read_csv(
            join(join(DATA_DIR, data_dir, mode), f), sep="\t", header=None
        )
        link_data.columns = ["entity1", "entity2"]
        entity1_id = link_data["entity1"].values.tolist()
        entity2_id = link_data["entity2"].values.tolist()
        for i, _ in enumerate(entity1_id):
            link[entity1_id[i]] = entity2_id[i]
            link[entity2_id[i]] = entity1_id[i]
        return link

    link = link_loader(args.language, args.data_dir)
    val_link = link_loader(args.language, args.data_dir, True)

    # Initialize model
    logging.info("Initializing model...")
    model = MyEmbedder(args, VOCAB_SIZE).to(device)
    _model = MyEmbedder(args, VOCAB_SIZE).to(device)
    _model.update(model)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # Training loop (simplified version of Trainer.train())
    def adjust_learning_rate(optimizer, epoch, lr):
        if (epoch + 1) % 10 == 0:
            lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def evaluate(step):
        logging.info("Evaluate at epoch {}...".format(step))
        print("Evaluate at epoch {}...".format(step))

        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        inverse_ids_2 = dict()
        with torch.no_grad():
            model.eval()
            for sample_id_1, (token_data_1, id_data_1) in tqdm(enumerate(eval_loader1)):
                entity_vector_1 = model(token_data_1).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1.squeeze().tolist())
                vector_1.append(entity_vector_1)

            for sample_id_2, (token_data_2, id_data_2) in tqdm(enumerate(eval_loader2)):
                entity_vector_2 = model(token_data_2).squeeze().detach().cpu().numpy()
                ids_2.extend(id_data_2.squeeze().tolist())
                vector_2.append(entity_vector_2)

        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx

        def cal_hit(v1, v2, link_data):
            source = [_id for _id in ids_1 if _id in link_data]
            target = np.array(
                [
                    (
                        inverse_ids_2[link_data[_id]]
                        if link_data[_id] in inverse_ids_2
                        else 99999
                    )
                    for _id in source
                ]
            )
            src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link_data]
            v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
            v2 = np.concatenate(tuple(v2), axis=0)
            index = faiss.IndexFlatL2(v2.shape[1])
            index.add(np.ascontiguousarray(v2))
            D, I = index.search(np.ascontiguousarray(v1), 10)
            hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
            hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)
            logging.info("#Entity: {}".format(len(source)))
            logging.info("Hit@1: {}".format(round(hit1, 3)))
            logging.info("Hit@10:{}".format(round(hit10, 3)))
            print("#Entity: {}".format(len(source)))
            print("Hit@1: {}".format(round(hit1, 3)))
            print("Hit@10:{}".format(round(hit10, 3)))
            return round(hit1, 3), round(hit10, 3)

        logging.info("========Validation========")
        print("========Validation========")
        hit1_valid, hit10_valid = cal_hit(vector_1, vector_2, val_link)
        logging.info("===========Test===========")
        print("===========Test===========")
        hit1_test, hit10_test = cal_hit(vector_1, vector_2, link)
        return hit1_valid, hit10_valid, hit1_test, hit10_test

    # Prepare batches
    all_data_batches = []
    for batch_id, (token_data, id_data) in enumerate(data_loader1):
        all_data_batches.append([1, token_data, id_data])
    for batch_id, (token_data, id_data) in enumerate(data_loader2):
        all_data_batches.append([2, token_data, id_data])
    random.shuffle(all_data_batches)

    del data_loader1
    del data_loader2

    neg_queue1 = None
    neg_queue2 = None

    logging.info("*** Evaluate at the very beginning ***")
    print("*** Evaluate at the very beginning ***")
    evaluate(0)

    best_hit1_valid_epoch = 0
    best_hit10_valid_epoch = 0
    best_hit1_test_epoch = 0
    best_hit10_test_epoch = 0
    best_hit1_valid = 0
    best_hit10_valid = 0
    best_hit1_valid_hit10 = 0
    best_hit10_valid_hit1 = 0
    best_hit1_test = 0
    best_hit10_test = 0
    best_hit1_test_hit10 = 0
    best_hit10_test_hit1 = 0
    record_hit1 = 0
    record_hit10 = 0
    record_epoch = 0
    record_batch_id = 0

    lr = args.lr

    for epoch in range(0, args.epoch):
        adjust_learning_rate(optimizer, epoch, lr)
        model.train()

        for batch_id, (language_id, token_data, id_data) in tqdm(
            enumerate(all_data_batches)
        ):
            pos_batch = None
            if language_id == 1:
                with torch.no_grad():
                    if neg_queue1 == None:
                        neg_queue1 = token_data.unsqueeze(0)
                    else:
                        neg_queue1 = torch.cat(
                            (neg_queue1, token_data.unsqueeze(0)), dim=0
                        )

                id_data = id_data.squeeze()

                if neg_queue1.shape[0] == args.queue_length + 1:
                    pos_batch = neg_queue1[0]
                    neg_queue1 = neg_queue1[1:]
                    neg_queue = neg_queue1
                else:
                    continue

            else:
                with torch.no_grad():
                    if neg_queue2 == None:
                        neg_queue2 = token_data.unsqueeze(0)
                    else:
                        neg_queue2 = torch.cat(
                            (neg_queue2, token_data.unsqueeze(0)), dim=0
                        )

                if neg_queue2.shape[0] == args.queue_length + 1:
                    pos_batch = neg_queue2[0]
                    neg_queue2 = neg_queue2[1:]
                    neg_queue = neg_queue2
                else:
                    continue

            optimizer.zero_grad()

            pos_1 = model(pos_batch)

            with torch.no_grad():
                _model.eval()
                pos_2 = _model(pos_batch)
                neg_shape = neg_queue.shape
                neg_queue_reshaped = neg_queue.reshape(
                    neg_shape[0] * neg_shape[1], neg_shape[2], -1
                )
                neg_value = _model(neg_queue_reshaped)

            contrastive_loss = model.contrastive_loss(pos_1, pos_2, neg_value)

            contrastive_loss.backward(retain_graph=True)
            optimizer.step()

            if batch_id == len(all_data_batches) - 1:
                logging.info(
                    "epoch: {} batch: {} loss: {}".format(
                        epoch,
                        batch_id,
                        contrastive_loss.detach().cpu().data / args.batch_size,
                    )
                )
                print(
                    "epoch: {} batch: {} loss: {}".format(
                        epoch,
                        batch_id,
                        contrastive_loss.detach().cpu().data / args.batch_size,
                    )
                )
                hit1_valid, hit10_valid, hit1_test, hit10_test = evaluate(
                    str(epoch) + ": batch " + str(batch_id)
                )

                if hit1_valid > best_hit1_valid:
                    best_hit1_valid = hit1_valid
                    best_hit1_valid_hit10 = hit10_valid
                    best_hit1_valid_epoch = epoch
                    record_epoch = epoch
                    record_batch_id = batch_id
                    record_hit1 = hit1_test
                    record_hit10 = hit10_test
                if hit10_valid > best_hit10_valid:
                    best_hit10_valid = hit10_valid
                    best_hit10_valid_hit1 = hit1_valid
                    best_hit10_valid_epoch = epoch
                    if hit1_valid == best_hit1_valid:
                        record_epoch = epoch
                        record_batch_id = batch_id
                        record_hit1 = hit1_test
                        record_hit10 = hit10_test

                if hit1_test > best_hit1_test:
                    best_hit1_test = hit1_test
                    best_hit1_test_hit10 = hit10_test
                    best_hit1_test_epoch = epoch
                if hit10_test > best_hit10_test:
                    best_hit10_test = hit10_test
                    best_hit10_test_hit1 = hit1_test
                    best_hit10_test_epoch = epoch

                logging.info(
                    "Test Hit@1(10)    = {}({}) at epoch {} batch {}".format(
                        hit1_test, hit10_test, epoch, batch_id
                    )
                )
                logging.info(
                    "Best Valid Hit@1  = {}({}) at epoch {}".format(
                        best_hit1_valid, best_hit1_valid_hit10, best_hit1_valid_epoch
                    )
                )
                logging.info(
                    "Best Valid Hit@10 = {}({}) at epoch {}".format(
                        best_hit10_valid, best_hit10_valid_hit1, best_hit10_valid_epoch
                    )
                )
                logging.info(
                    "Test @ Best Valid = {}({}) at epoch {} batch {}".format(
                        record_hit1, record_hit10, record_epoch, record_batch_id
                    )
                )
                logging.info(
                    "Best Test  Hit@1  = {}({}) at epoch {}".format(
                        best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch
                    )
                )
                logging.info(
                    "Best Test  Hit@10 = {}({}) at epoch {}".format(
                        best_hit10_test, best_hit10_test_hit1, best_hit10_test_epoch
                    )
                )
                logging.info("====================================")

                print(
                    "Test Hit@1(10)    = {}({}) at epoch {} batch {}".format(
                        hit1_test, hit10_test, epoch, batch_id
                    )
                )
                print(
                    "Best Valid Hit@1  = {}({}) at epoch {}".format(
                        best_hit1_valid, best_hit1_valid_hit10, best_hit1_valid_epoch
                    )
                )
                print(
                    "Best Valid Hit@10 = {}({}) at epoch {}".format(
                        best_hit10_valid, best_hit10_valid_hit1, best_hit10_valid_epoch
                    )
                )
                print(
                    "Test @ Best Valid = {}({}) at epoch {} batch {}".format(
                        record_hit1, record_hit10, record_epoch, record_batch_id
                    )
                )
                print(
                    "Best Test  Hit@1  = {}({}) at epoch {}".format(
                        best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch
                    )
                )
                print(
                    "Best Test  Hit@10 = {}({}) at epoch {}".format(
                        best_hit10_test, best_hit10_test_hit1, best_hit10_test_epoch
                    )
                )
                print("====================================")

            _model.update(model)

    logging.info("Training completed!")
    logging.info(f"Final results - Test @ Best Valid = {record_hit1}({record_hit10})")
