from settings import *
import csv
import pandas as pd
import torch
import pickle


class CustomRawNeighbors:
    """Loader for custom datasets - compatible with DBP15KRawNeighbors interface."""

    def __init__(self, language, doc_id, data_dir="custom"):
        self.language = language
        self.doc_id = doc_id
        self.data_dir = data_dir
        self.path = join(DATA_DIR, self.data_dir, self.language)
        self.id_entity = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.load()
        self.id_neighbors_loader()
        self.get_center_adj()

    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + ".pkl"), "rb") as f:
            self.id_entity = pickle.load(f)

        # Ensure embeddings are in correct format: id -> [embedding]
        # Handle both old format (id -> embedding) and new format (id -> [embedding])
        for entity_id, embedding in self.id_entity.items():
            if isinstance(embedding, list) and len(embedding) > 0:
                # Check if it's already wrapped or is a flat embedding
                if isinstance(embedding[0], (int, float)):
                    # Old format: flat embedding, wrap it
                    self.id_entity[entity_id] = [embedding]

    def id_neighbors_loader(self):
        data = pd.read_csv(
            join(self.path, "triples_" + self.doc_id), header=None, sep="\t"
        )
        data.columns = ["head", "relation", "tail"]

        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            head_str = self.id_entity[int(row["head"])][0]
            tail_str = self.id_entity[int(row["tail"])][0]

            if not int(row["head"]) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row["head"])] = [head_str]
            if not tail_str in self.id_neighbors_dict[int(row["head"])]:
                self.id_neighbors_dict[int(row["head"])].append(tail_str)

            if not int(row["tail"]) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row["tail"])] = [tail_str]
            if not head_str in self.id_neighbors_dict[int(row["tail"])]:
                self.id_neighbors_dict[int(row["tail"])].append(head_str)

    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):
        for k, v in self.id_neighbors_dict.items():
            if len(v) < NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.id_neighbors_dict[k] = v + [[0] * LaBSE_DIM] * (
                    NEIGHBOR_SIZE - len(v)
                )
            else:
                self.id_adj_tensor_dict[k] = self.get_adj(NEIGHBOR_SIZE)
                self.id_neighbors_dict[k] = v[:NEIGHBOR_SIZE]
