import json
import random
from tqdm import tqdm
import os

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

from pyserini.search import LuceneSearcher

logger = get_logger(__name__, log_level="INFO")

E5_EMBEDDING_PROMPTS = {
    "allnli": [
        "Given a premise, retrieve a hypothesis that is entailed by the premise",
        "Retrieve semantically similar text",
    ],
    "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query",
    "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "quora_duplicates": [
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Find questions that have the same meaning as the input question",
    ],
    "squad": "Retrieve Wikipedia passages that answer the question",
    "trivia_qa": "Retrieve Wikipedia passages that answer the question",
}

class E5Data(Dataset):
    def __init__(
        self,
        dataset_name: str = "E5",
        split: str = "validation",
        file_path: str = "cache/echo-data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        n_ic_examples: int = 0,
        retrieve_queries: bool = False,
        use_negatives: bool = False,
        use_instructions: bool = True,
        dataset_size: int = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.n_ic_examples = n_ic_examples
        self.retrieve_queries = retrieve_queries
        self.use_negatives = use_negatives
        self.use_instructions = use_instructions
        self.dataset_size = dataset_size

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        print(f"Loading E5 data from {file_path}...")

        data_map = {}
        all_samples = []
        separated_samples = {}
        id_ = 0
        new_id_ = 0
        for dataset in E5_EMBEDDING_PROMPTS:
            separated_samples[dataset] = []
            print(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                dataset_samples = f.readlines()

            dataset_samples = [json.loads(d) for d in dataset_samples]

            for i, sample in enumerate(dataset_samples):
                query = sample["query"]
                pos = sample["positive"]
                neg = sample["negative"]

                separated_samples[dataset].append(
                    DataSample(
                        id_=id_,
                        query=query,
                        query_flipped=None,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                        
                    )
                )
                id_ += 1

        for dataset in E5_EMBEDDING_PROMPTS:
            print(f"Constructing IC queries for {dataset}...")
            print(f"Using {self.n_ic_examples} IC examples.")

            if self.retrieve_queries:
                index_dir = os.path.join(os.environ.get("TRANSFORMERS_CACHE", "temp_train_index"), f"temp_indexes_", dataset)
                print(f"Retrieving queries for {dataset}...")
                if not os.path.exists(index_dir):
                    searcher = self.encode_ic_data(separated_samples[dataset], index_dir=index_dir)
                else:
                    searcher = LuceneSearcher(index_dir)
            else:
                searcher = None
            
            if self.dataset_size is None:
                print(f"Using all samples for {dataset}.")
                dataset_samples = separated_samples[dataset]
            else:
                ratio = len(separated_samples[dataset])/sum([len(separated_samples[dataset]) for dataset in separated_samples])
                n_samples = int(ratio * self.dataset_size)
                dataset_samples = random.sample(separated_samples[dataset], n_samples if len(separated_samples[dataset]) > n_samples else len(separated_samples[dataset]))

            for i, sample in enumerate(tqdm(dataset_samples)):
                instruction = (
                        E5_EMBEDDING_PROMPTS[dataset]
                        if isinstance(E5_EMBEDDING_PROMPTS[dataset], str)
                        else E5_EMBEDDING_PROMPTS[dataset][i % 2]
                    )
                data_map[dataset].append(new_id_)
                new_query, new_query_flipped = self.construct_ic_query(sample, self.n_ic_examples, searcher=searcher, ic_data=separated_samples[dataset], use_negatives=self.use_negatives, instruction=instruction if self.use_instructions else None)
                pos = sample.positive
                neg = sample.negative
                if dataset in [
                    "allnli_split2",
                    "quora_duplicates_split1",
                    "quora_duplicates_split2",
                ]:
                    pos = (
                        f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                        + self.separator
                        + pos
                    )
                    neg = (
                        f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                        + self.separator
                        + neg
                    )
                else:
                    pos = self.separator + pos
                    neg = self.separator + neg

                all_samples.append(
                    DataSample(
                        id_=new_id_,
                        query=new_query,
                        query_flipped=new_query_flipped,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                new_id_ += 1

        # combine split1 and split2
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset.replace("_split1", "").replace("_split2", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        print(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    print(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        print(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative, sample.query_flipped], label=1.0
            )
        elif self.split == "validation":
            assert False, "E5Data does not have a validation split."
