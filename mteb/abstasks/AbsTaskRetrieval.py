from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from time import time
import time as time_module
from typing import Any

from datasets import Features, Value, load_dataset

from mteb.abstasks.TaskMetadata import HFSubset

from ..evaluation.evaluators import RetrievalEvaluator
from ..load_results.mteb_results import ScoresDict
from .AbsTask import AbsTask, DescriptiveStatistics

import random
import numpy as np
from pyserini.search import SimpleSearcher
from pyserini.index.lucene import LuceneIndexer


logger = logging.getLogger(__name__)


def split_dict(data, test_size=0.2):
    # Convert dictionary keys to a list
    keys = list(data.keys())
    
    # Shuffle keys
    random.shuffle(keys)
    
    # Determine the split point
    split_point = int(len(keys) * (1 - test_size))
    
    # Split the keys into train and test
    train_keys = keys[:split_point]
    test_keys = keys[split_point:]
    
    # Create train and test dictionaries
    train_dict = {key: data[key] for key in train_keys}
    test_dict = {key: data[key] for key in test_keys}
    
    return train_dict, test_dict


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    def __init__(
        self,
        hf_repo: str | None = None,
        hf_repo_qrels: str | None = None,
        data_folder: str | None = None,
        prefix: str | None = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
        streaming: bool = False,
        keep_in_memory: bool = False,
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.hf_repo = hf_repo
        if hf_repo:
            # By default fetch qrels from same repo not a second repo with "-qrels" like in original
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
            # data folder would contain these files:
            # (1) fiqa/corpus.jsonl  (format: jsonlines)
            # (2) fiqa/queries.jsonl (format: jsonlines)
            # (3) fiqa/qrels/test.tsv (format: tsv ("\t"))
            if prefix:
                query_file = prefix + "-" + query_file
                qrels_folder = prefix + "-" + qrels_folder

            self.corpus_file = (
                os.path.join(data_folder, corpus_file) if data_folder else corpus_file
            )
            self.query_file = (
                os.path.join(data_folder, query_file) if data_folder else query_file
            )
            self.qrels_folder = (
                os.path.join(data_folder, qrels_folder) if data_folder else None
            )
            self.qrels_file = qrels_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load(
        self, split="test"
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        if not self.hf_repo:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self._load_qrels(split)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(
                self.hf_repo,
                "corpus",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                self.hf_repo,
                "queries",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        queries_ds = next(iter(queries_ds.values()))  # get first split
        queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds

    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )[split]
        else:
            qrels_ds = load_dataset(
                "csv",
                data_files=self.qrels_file,
                delimiter="\t",
                keep_in_memory=self.keep_in_memory,
            )
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


class RetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_queries: number of samples in the dataset
        average_document_length: Average length of documents
        average_query_length: Average length of queries
        num_documents: Number of documents
        average_relevant_docs_per_query: Average number of relevant documents per query
    """

    num_queries: int
    average_document_length: float
    average_query_length: float
    num_documents: int
    average_relevant_docs_per_query: float


class AbsTaskRetrieval(AbsTask):
    """Abstract class for retrieval experiments.

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, Union[str, List[str]]]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, List[str]]] for conversations
        E.g. {"test": {"q1": "query"}}
        or {"test": {"q1": ["turn1", "turn2", "turn3"]}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}}
    """

    ignore_identical_ids: bool = False

    def __init__(self, random_ic_prompt=False, doc_only=False, **kwargs):
        super().__init__(**kwargs)
        self.random_ic_prompt = random_ic_prompt
        self.doc_only = doc_only

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = HFDataLoader(
                hf_repo=dataset_path,
                hf_repo_qrels=hf_repo_qrels,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # Conversion from DataSet
            queries = {query["id"]: query["text"] for query in queries}

            if self.n_ic_examples > 0:
                index_dir = os.path.join(os.environ.get("TRANSFORMERS_CACHE", "temp_index_eval"), "temp_index_eval", dataset_path.split("/")[-1])
                queries = self.encode_queries_with_ic_data(queries, index_dir, qrels, **kwargs)

            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def evaluate(
        self,
        model,
        split: str = "test",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        retriever = RetrievalEvaluator(
            retriever=model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
            )
        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
        start_time = time()
        results = retriever(corpus, queries)
        end_time = time()
        logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        save_predictions = kwargs.get("save_predictions", False)
        export_errors = kwargs.get("export_errors", False)
        if save_predictions or export_errors:
            output_folder = Path(kwargs.get("output_folder", "results"))
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

        if save_predictions:
            top_k = kwargs.get("top_k", None)
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[:top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs.items()
            },
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs_mrr.items()
            },
        }
        self._add_main_score(scores)

        if export_errors:
            errors = {}

            top_k = kwargs.get("top_k", 1)
            if not save_predictions and top_k == 1:
                for qid in results.keys():
                    doc_scores = results[qid]
                    sorted_docs = sorted(
                        doc_scores.items(), key=lambda x: x[1], reverse=True
                    )[:top_k]
                    results[qid] = dict(sorted_docs)
            for qid, retrieved_docs in results.items():
                expected_docs = relevant_docs[qid]
                false_positives = [
                    doc for doc in retrieved_docs if doc not in expected_docs
                ]
                false_negatives = [
                    doc for doc in expected_docs if doc not in retrieved_docs
                ]
                if false_positives or false_negatives:
                    errors[qid] = {
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }

            errors_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_errors.json"
            )
            with open(errors_save_path, "w") as f:
                json.dump(errors, f)

        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RetrievalDescriptiveStatistics:
        if hf_subset:
            queries = self.queries[hf_subset][split]
            corpus = self.corpus[hf_subset][split]
            relevant_docs = self.relevant_docs[hf_subset][split]
        elif compute_overall:
            queries = {}
            corpus = {}
            relevant_docs = {}
            for hf_subset in self.metadata.eval_langs:
                queries.update(process_docs(self.queries, hf_subset, split))
                corpus.update(process_docs(self.corpus, hf_subset, split))
                relevant_docs.update(
                    process_relevant_docs(self.relevant_docs, hf_subset, split)
                )
        else:
            queries = self.queries[split]
            corpus = self.corpus[split]
            relevant_docs = self.relevant_docs[split]

        query_len, doc_len = calculate_length(queries, corpus)
        num_documents = len(corpus)
        num_queries = len(queries)

        # number of qrels that are not 0
        num_qrels_non_zero = sum(
            sum(1 for doc_id in docs if docs[doc_id] != 0)
            for docs in relevant_docs.values()
        )
        qrels_per_doc = num_qrels_non_zero / len(relevant_docs) if num_queries else 0
        return RetrievalDescriptiveStatistics(
            average_document_length=doc_len,
            average_query_length=query_len,
            num_documents=num_documents,
            num_queries=num_queries,
            average_relevant_docs_per_query=qrels_per_doc,
        )
    
    def encode_queries_with_ic_data(self, queries, index_dir, qrels, **kwargs):
        """
        Encode queries with in-context data.

        Args:
        queries: list of queries [str]
        ic_data: list of in-context data [list of dicts]
        searcher: SimpleSearcher instance with pre-encoded index
        top_n: number of similar in-context examples to select

        Returns:
        encoded_queries: list of encoded queries [np.ndarray]
        """
        ic_corpus, ic_queries, relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        if "RAR" in dataset_path: 
            modified_dataset_path = dataset_path
            if "piqa" in dataset_path: modified_dataset_path = "../data/piqa"
            elif "siqa" in dataset_path: modified_dataset_path = "../data/siqa"
            elif "TempReason-l1" in dataset_path: modified_dataset_path = "../data/TempReason"
            elif "quail" in dataset_path: modified_dataset_path = "../data/quail"
            elif "ARC-Challenge" in dataset_path: modified_dataset_path = "../data/ARC-Challenge"
            elif "hellaswag" in dataset_path: modified_dataset_path = "../data/hellaswag"
            elif "alphanli" in dataset_path: modified_dataset_path = "../data/alphanli"
            elif "winogrande" in dataset_path: modified_dataset_path = "../data/winogrande"
            print(dataset_path, modified_dataset_path)
            ic_corpus, ic_queries, ic_qrels = HFDataLoader(data_folder=modified_dataset_path, streaming=False, keep_in_memory=False).load(split="train")
        else:
            try:
                for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
                    ic_corpus, ic_queries, ic_qrels = HFDataLoader(
                        hf_repo=dataset_path,
                        hf_repo_qrels=hf_repo_qrels,
                        streaming=False,
                        keep_in_memory=False,
                    ).load(split="train")
            except:
                try:
                    for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
                        ic_corpus, ic_queries, ic_qrels = HFDataLoader(
                            hf_repo=dataset_path,
                            hf_repo_qrels=hf_repo_qrels,
                            streaming=False,
                            keep_in_memory=False,
                        ).load(split="dev")
                except:
                    dataset_path = "BeIR/" + dataset_path.rstrip("/").split("/")[-1].split("-")[0] + "-generated-queries"
                    print(f"Loading in-context data from {dataset_path}...")
                    ic_data = {}
                    data = load_dataset(dataset_path, split="train")
                    data_dict = {str(i): dict(row) for i, row in enumerate(data)}

                    for i, row in data_dict.items():
                        ic_sample = {}
                        ic_sample["id"] = str(row["_id"])
                        ic_sample["question"] = row["query"]
                        ic_sample["positive_ctxs"] = []
                        ic_sample["negative_ctxs"] = []
                        pos_ctx = {}
                        pos_ctx["title"] = row["title"]
                        pos_ctx["text"] = row["text"]
                        neg_ctx = {}
                        random_idx = np.random.choice(len(data_dict), 1)[0]
                        neg_ctx["title"] = data_dict[str(random_idx)]["title"]
                        neg_ctx["text"] = data_dict[str(random_idx)]["text"]
                        ic_sample["positive_ctxs"].append(pos_ctx)
                        ic_sample["negative_ctxs"].append(neg_ctx)
                        ic_data[str(i)] = ic_sample

                    if self.fixed_ic_prompt:
                        print("Using fixed in-context prompts...")
                        encoded_queries = self.process_ic_data(queries, ic_data, searcher=None, top_n=self.n_ic_examples)
                    else:
                        searcher = self.encode_ic_data(ic_data, index_dir=index_dir)
                        encoded_queries = self.process_ic_data(queries, ic_data, searcher=searcher, top_n=self.n_ic_examples)
                    
                    return encoded_queries

        ic_corpus = ic_corpus.to_pandas().set_index("id").to_dict(orient="index")
        ic_queries = ic_queries.to_pandas().set_index("id").to_dict(orient="index")
        ic_queries = {key: ic_queries[key]["text"] for key in ic_queries.keys()}

        ic_data = self.construct_ic_dataset(ic_corpus, ic_queries, ic_qrels)
        
        if self.random_ic_prompt:
            print("Using random in-context prompts...")
            encoded_queries = self.process_ic_data(queries, ic_data, searcher=None, top_n=self.n_ic_examples)
        else:
            searcher = self.encode_ic_data(ic_data, index_dir=index_dir)
            encoded_queries = self.process_ic_data(queries, ic_data, searcher=searcher, top_n=self.n_ic_examples)

        return encoded_queries
    
    def construct_ic_dataset(self, ic_corpus, ic_queries, ic_qrels):
        ic_data = {}
        for query_id in ic_queries.keys():
            query = ic_queries[query_id]
            rels = ic_qrels[query_id]
            ic_example = {}
            max_rel_doc = {}
            max_value = max(rels.values())
            max_rel_doc_ids = [key for key, value in rels.items() if value == max_value]
            max_rel_doc_id = random.choice(max_rel_doc_ids)
            min_value = min(rels.values())
            min_rel_doc_ids = [key for key, value in rels.items() if value == min_value]
            min_rel_doc_id = random.choice(min_rel_doc_ids)
            ic_example["id"] = str(query_id)
            ic_example["positive_ctxs"] = []
            max_rel_doc["title"] = ic_corpus[max_rel_doc_id]["title"]
            max_rel_doc["text"] = ic_corpus[max_rel_doc_id]["text"]
            ic_example["positive_ctxs"].append(max_rel_doc)
            ic_example["negative_ctxs"] = []
            min_rel_doc = {}
            min_rel_doc["title"] = ic_corpus[min_rel_doc_id]["title"]
            min_rel_doc["text"] = ic_corpus[min_rel_doc_id]["text"]
            ic_example["negative_ctxs"].append(min_rel_doc)
            ic_example["question"] = query
            ic_data[str(query_id)] = ic_example
        return ic_data
    
    def encode_ic_data(self, ic_data, index_dir='temp_index'):
        """
        Encode the in-context data using pyserini and keep the index in memory.

        Args:
        ic_data: list of in-context data [list of dicts]

        Returns:
        searcher: SimpleSearcher instance with in-memory index
        ic_questions: List of questions from the in-context data
        """
        
        documents = [{"id": f"doc{idx}", "contents": item["question"]} for idx, item in ic_data.items()]

        os.makedirs(index_dir, exist_ok=True)
        max_wait_time = 60 

        start_time = time()

        while True:
            try:
                indexer = LuceneIndexer(index_dir=index_dir, threads=1)
                indexer.add_batch_dict(documents)
                indexer.close()
                
                break
            except Exception as e:
                if "Lock obtain timed out" in str(e):
                    elapsed_time = time() - start_time
                    if elapsed_time >= max_wait_time:
                        raise TimeoutError(f"Exceeded maximum wait time of {max_wait_time} seconds for lock release.")
                    print("Write lock detected, waiting for release...")
                    time_module.sleep(1)
                else:
                    raise e 

        searcher = SimpleSearcher(index_dir)
        
        return searcher

    def process_ic_data(self, queries, ic_data, searcher=None, top_n=5):
        """
        Add in-context data to the queries based on similarity using BM25.

        Args:
        queries: dict of queries [dict]
        ic_data: dict of in-context data [dict of dicts]
        searcher: SimpleSearcher instance with pre-encoded index
        top_n: number of similar in-context examples to select

        Returns:
        new_queries: list of queries with context data added
        """
        new_queries = {}
        prefix = ""
        for key, query in queries.items():
            new_query = prefix
            if top_n > 0:
                if searcher is None:
                    keys = list(ic_data.keys())
                    random.shuffle(keys)
                    top_idxs = keys[:top_n]
                else:
                    hits = searcher.search(query, k=top_n+100)
                    hits = [hit for hit in hits if hit.docid.split('doc')[-1]]
                    top_idxs = [hit.docid.split('doc')[-1] for hit in hits]
                
                top_idxs = top_idxs[:top_n]
                top_idxs = top_idxs[::-1]
                for k, idx in enumerate(top_idxs):
                    idx = str(idx)
                    question = ic_data[idx]["question"]
                    positive_ctx = ic_data[idx]["positive_ctxs"][0]["title"] + ". " + ic_data[idx]["positive_ctxs"][0]["text"]
                    if k == 0:
                        new_query += f"{question}\nPositive Document: {positive_ctx}\n"
                    else:
                        if self.doc_only:
                            new_query += f"Positive Document: {positive_ctx}\n"
                        else:
                            new_query += f"Query: {question}\nPositive Document: {positive_ctx}\n"
            new_query += "Query: " + query
            new_queries[key] = new_query
        return new_queries


def calculate_length(
    queries: dict[str, str], corpus: dict[str, str]
) -> tuple[float, float]:
    queries_lens = []
    doc_lens = []
    for query in queries.values():
        queries_lens.append(len(query))

    for doc in corpus.values():
        if isinstance(doc, dict):
            doc_lens.append(len(doc.get("title", "")) + len(doc["text"]))
        else:
            doc_lens.append(len(doc))

    doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    query_len = sum(queries_lens) / len(queries_lens) if queries_lens else 0
    return query_len, doc_len


def process_docs(
    collection: dict[str, dict[str, dict[str, str] | str]], hf_subset: str, split: str
) -> dict[str, str]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return {
        f"{split}_{hf_subset}_{k}": v for k, v in collection[hf_subset][split].items()
    }


def process_relevant_docs(
    collection: dict[str, dict[str, dict[str, dict[str, int]]]],
    hf_subset: str,
    split: str,
) -> dict[str, dict[str, int]]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return_collection = {}
    for query_id, relevant in collection[hf_subset][split].items():
        return_collection[f"{split}_{hf_subset}_{query_id}"] = {
            f"{split}_{hf_subset}_{doc_id}": value for doc_id, value in relevant.items()
        }
    return return_collection
