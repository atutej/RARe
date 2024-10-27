"""Merges CQADupstack subset results
Usage: python merge_cqadupstack.py path_to_results_folder
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_LIST_CQA = [
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
]

NOAVG_KEYS = [
    "evaluation_time",
    "mteb_version",
    "task_name",
    "dataset_revision",
    "hf_subset",
    "languages",
]


results_folder = sys.argv[1]
# Ensure at least 1 character btw CQADupstack & Retrieval
files = glob.glob(f'{results_folder.rstrip("/")}/CQADupstack*?*Retrieval.json')

logger.info(f"Found CQADupstack files: {files}")


if len(files) != len(TASK_LIST_CQA):
    logger.warning(
        f"Got {len(files)}, but expected {len(TASK_LIST_CQA)} files. Missing: {set(TASK_LIST_CQA) - set([x.split('/')[-1].split('.')[0] for x in files])}; Too much: {set([x.split('/')[-1].split('.')[0] for x in files]) - set(TASK_LIST_CQA)}"
    )
    exit(1)


all_results = {"scores": {}}
for file_name in files:
    with open(file_name, "r", encoding="utf-8") as f:
        results = json.load(f)

        for key, value in results.items():
            if key in ["scores", "task_name"]:
                continue
            all_results[key] = value

        for split, split_results in results['scores'].items():
            if split not in ("train", "validation", "dev", "test"):
                all_results['scores'][split] = split_results
                continue
            for split_result in split_results:
                all_results['scores'].setdefault(split, [{}])
                for metric, score in split_result.items():
                    all_results['scores'][split][0].setdefault(metric, 0)
                    if metric == "evaluation_time":
                        score = all_results['scores'][split][0][metric] + score
                    elif metric not in NOAVG_KEYS:
                        score = all_results['scores'][split][0][metric] + score * 1 / len(
                            TASK_LIST_CQA
                        )
                    all_results['scores'][split][0][metric] = score
all_results["task_name"] = "CQADupstackRetrieval"

logger.info("Saving ", all_results)
with open(
    os.path.join(results_folder, "CQADupstackRetrieval.json"), "w", encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=4)