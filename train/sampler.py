from temp.taxo import TaxStruct
from torch.utils.data import Dataset as TorchDataset
import transformers
import torch
import numpy as np
import random


class Sampler:
    def __init__(self, tax_graph: TaxStruct):
        self._tax_graph = tax_graph
        self._nodes = list(self._tax_graph.nodes.keys())

    def sampling(self):
        margins = []
        pos_paths = []
        neg_paths = []
        for node, path in self._tax_graph.node2path.items():
            if node == self._tax_graph.root:
                continue
            while True:
                neg_node = random.choice(self._nodes)
                if neg_node != path[1] and neg_node != node:
                    break
            pos_paths.append(path)
            neg_path = [node] + self._tax_graph.node2path[neg_node]
            neg_paths.append(neg_path)
            margins.append(self.margin(path, neg_path))
        return pos_paths, neg_paths, margins

    @staticmethod
    def margin(path_a, path_b):
        com = len(set(path_a).intersection(set(path_b)))
        return max(min((abs(len(path_a) - com) + abs(len(path_b) - com)) / com, 2), 0.5)


class PhiDataset(torch.utils.data.Dataset):

    def __init__(self, sampler, tokenizer, word2des, padding_max=512, margin_beta=0.1):
        self._sampler = sampler
        self._word2des = word2des
        self._padding_max = padding_max
        self._margin_beta = margin_beta
        self._tokenizer = tokenizer

        if self._sampler is not None:
            self._pos_paths, self._neg_paths, self._margins = self._sampler.sampling()

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos_path = self._pos_paths[item]
        neg_path = self._neg_paths[item]
        margin = self._margins[item]

        pos_ids, pos_attn_masks = self.encode_path_as_prompt(pos_path)
        neg_ids, neg_attn_masks = self.encode_path_as_prompt(neg_path)

        return dict(
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_attn_masks=pos_attn_masks,
            neg_attn_masks=neg_attn_masks,
            margin=torch.FloatTensor([margin * self._margin_beta])
        )

    def encode_path_as_prompt(self, path):

        query_term = path[0]
        query_definition = self._word2des.get(query_term, [""])[0] 
        
        # Build path string: path[0] is query, path[1:] is ancestors from root (TaxStruct builds path from child to root, so reversed it)
        path_str = " → ".join(reversed(path[1:])) if len(path) > 1 else "Root"

        user_prompt = f'''Query: {query_term}\n
  Definition: {query_definition}\n
  Candidate path: {path_str}\n
  Task: Rate the plausibility that this query belongs to the given path. Output a real-valued score.'''

        messages = [
            {"role": "system", "content": "You are an expert taxonomist that rates the plausibility of a concept belonging to a hierarchical path."},
            {"role": "user", "content": user_prompt}
        ]

        # I have set add_generation_prompt = False because I am encoding for representation, not generating a response.
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False 
        )
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            max_length=self._padding_max,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)
    def resample(self):
        """Resample positive and negative paths for new epoch"""
        if self._sampler is not None:
            self._pos_paths, self._neg_paths, self._margins = self._sampler.sampling()
