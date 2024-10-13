import os
import random

from dataclasses import dataclass
from typing import Union, List

import torch

from pyserini.index.lucene import LuceneIndexer
from pyserini.search import SimpleSearcher, FaissSearcher
import faiss
from sentence_transformers import SentenceTransformer

from collections import defaultdict


@dataclass
class DataSample:
    id_: int
    query: str
    positive: str
    query_flipped: str = None
    negative: str = None
    task_name: str = None
    flipped: bool = False


class TrainSample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
        self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0
    ):
        """
        Creates one TrainSample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<TrainSample> label: {}, texts: {}".format(
            str(self.label), "; ".join(self.texts)
        )

class Dataset(torch.utils.data.Dataset):
    def load_data(self, file_path: str = None):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def encode_ic_data(self, ic_data, index_dir='temp_index'):
        """
        Encode the in-context data using pyserini and keep the index in memory.

        Args:
        ic_data: list of in-context data [list of dicts]

        Returns:
        searcher: SimpleSearcher instance with in-memory index
        ic_questions: List of questions from the in-context data
        """
        ic_questions = [item.query for item in ic_data]
        
        documents = [{"id": f"doc{idx}", "contents": question} for idx, question in enumerate(ic_questions)]
        
        os.makedirs(index_dir, exist_ok=True)
        indexer = LuceneIndexer(index_dir=index_dir, threads=1)
        
        indexer.add_batch_dict(documents)
        indexer.close()
        
        searcher = SimpleSearcher(index_dir)
        
        return searcher
    

    def construct_ic_query(self, sample, n_ic_examples, searcher=None, ic_data=None, use_negatives=False, instruction=None):
        query = sample.query

        flipped = sample.flipped

        if ic_data is None:
            ic_data = [example for example in self.ic_data if example.flipped == flipped]

        if searcher:
            hits = searcher.search(query, k=n_ic_examples+1)
            ic_examples = [ic_data[int(hit.docid.split('doc')[-1])] for hit in hits]
            ic_examples = ic_examples[::-1]
            ic_examples = [example for example in ic_examples if example.query != query]

        else:
            ic_examples = random.sample(ic_data, n_ic_examples)

        if instruction:
            ic_query = f"Instruct: {instruction}\n"
            ic_query_flipped = f"Instruct: {instruction}\n"
        else:
            ic_query = ""
            ic_query_flipped = ""
        
        for example in ic_examples:
            if use_negatives:
                ic_query += f"Query: {example.query}\nPositive Document: {example.positive}\nNegative Document: {example.negative}\n\n"
            else:
                ic_query += f"Query: {example.query}\nPositive Document: {example.positive}\n\n"
            
        ic_query += self.separator
        ic_query += f"Query: {query}"
        return ic_query