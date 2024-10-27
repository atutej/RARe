import requests
import zipfile
import os
import json
import csv
import collections
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm
import argparse
import requests
import shutil

parser = argparse.ArgumentParser(description='download RAR-b benchmark')
parser.add_argument('--download_path', default="../data", type=str, help='download directory')
args = parser.parse_args()

rarb_datasets_with_labels = ["alphanli", "piqa", "siqa"]
rarb_datasets_without_labels = ["ARC-Challenge", "hellaswag", "quail", "TempReason-l1", "winogrande"]
rarb_dataset_download_links = {
    "alphanli": "https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip",
    "ARC-Challenge": "https://hub.oxen.ai/api/repos/datasets/ARC-Challenge/file/main/arc_challenge_train.jsonl",
    "hellaswag": "https://raw.github.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "piqa": "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
    "quail": "https://raw.github.com/text-machine-lab/quail/master/quail_v1.3/json/train.jsonl",
    "siqa": "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
    "TempReason-l1": "",
    "winogrande": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"
}

def download_and_unzip(dataset_name, download_path):
    print(f"######## Downloading {dataset_name} ########")
    dataset_path = os.path.join(download_path, dataset_name)
    snapshot_download(repo_id="RAR-b/"+dataset_name, repo_type="dataset", local_dir=dataset_path)

    if dataset_name in ["ARC-Challenge", "hellaswag", "quail"]:
        response = requests.get(rarb_dataset_download_links[dataset_name])
        if response.status_code == 200:
            with open(os.path.join(dataset_path, 'train.jsonl'), "wb") as f: f.write(response.content)
        else: print(f"Failed to download file: {response.status_code}")
    elif dataset_name == "TempReason-l1":
        hf_hub_download(repo_id="tonytan48/TempReason", filename="train_l1.json", local_dir=dataset_path, repo_type="dataset")
        os.rename(os.path.join(dataset_path, "train_l1.json"), os.path.join(dataset_path, "train.json"))
    else:
        response = requests.get(rarb_dataset_download_links[dataset_name])
        zip_file_path = os.path.join(download_path, 'temp.zip')

        with open(zip_file_path, 'wb') as zip_file:
            zip_file.write(response.content)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            os.makedirs(dataset_path, exist_ok=True)

            for member in zip_ref.namelist():
                zip_ref.extract(member, dataset_path)
                extracted_path = os.path.join(dataset_path, member)
                if os.path.isdir(extracted_path): os.rmdir(extracted_path)
                else: os.rename(extracted_path, os.path.join(dataset_path, os.path.basename(member)))

        if dataset_name == "winogrande": os.rename(os.path.join(dataset_path, "train_xl.jsonl"), os.path.join(dataset_path, "train.jsonl"))
        os.remove(zip_file_path)

    print(f"Downloaded and unzipped to: {dataset_path}")

def parse_dataset_with_labels(dataset_name, download_path):
    print(f"######## Parsing {dataset_name} ########")
    dataset_path = os.path.join(download_path, dataset_name)

    data_list = []
    with open(dataset_path+"/train.jsonl", "r") as f1:
        for line in f1: data_list.append(json.loads(line))

    label_list = []
    with open(dataset_path+"/train-labels.lst", "r") as f2:
        for line in f2: label_list.append(line.strip())

    corpus_inv_dict = {}
    with open(dataset_path+"/corpus.jsonl", "r") as f4:
        for line in f4:
            corpus_line = json.loads(line)
            corpus_inv_dict[corpus_line["text"]] = corpus_line["_id"]

    assert len(data_list) == len(label_list)

    parsed_data = []
    i = 0
    if dataset_name == "alphanli":
        instances = []
        for label, data in zip(tqdm(label_list), data_list):
            document = corpus_inv_dict[data["hyp1"] if int(label) == 1 else data["hyp2"]]
            if "Start: " + data["obs1"] + " End: " + data["obs2"] + document in instances: continue
            data_dict = collections.OrderedDict()
            data_dict["_id"] = "alpha-tq-"+str(i)
            data_dict["text"] = "Start: " + data["obs1"] + " End: " + data["obs2"]
            data_dict["metadata"] = {}
            instances.append(data_dict["text"] + document)
            parsed_data.append((data_dict, document))
            i += 1
    else:
        for label, data in zip(tqdm(label_list), data_list):
            data_dict = collections.OrderedDict()

            if dataset_name == "piqa":
                data_dict["_id"] = "piqa-tq-"+str(i)
                data_dict["text"] = data["goal"]
                document = corpus_inv_dict[data["sol1"] if int(label) == 0 else data["sol2"]]
            elif dataset_name == "siqa":
                data_dict["_id"] = "siqa-tq-"+str(i)
                data_dict["text"] = "Context: "+data["context"]+" Question: "+data["question"]
                document = corpus_inv_dict[data["answerA"] if int(label) == 1 else data["answerB"] if int(label) == 2 else data["answerC"]]

            data_dict["metadata"] = {}
            parsed_data.append((data_dict, document))
            i += 1
    
    with open(dataset_path+"/queries.jsonl", "a") as f:
        for data_dict, _ in parsed_data:
            f.write( json.dumps(data_dict, ensure_ascii=False) + "\n" )

    with open(dataset_path+"/qrels/train.tsv", "w") as f3:
        tw = csv.writer(f3, delimiter='\t')
        tw.writerow(['query-id', 'corpus-id', 'score'])
        for data_dict, document in parsed_data:
            tw.writerow([data_dict["_id"], document, 1])
    
    print(f"Parsed {dataset_name}: {str(len(parsed_data))} training examples")

def parse_dataset_without_labels(dataset_name, download_path):
    print(f"######## Parsing {dataset_name} ########")
    dataset_path = os.path.join(download_path, dataset_name)

    data_list = []
    if dataset_name == "TempReason-l1": 
        with open(dataset_path+"/train.json", "r") as f1:
            for line in f1: data_list.append(json.loads(line))
    else:
        with open(dataset_path+"/train.jsonl", "r") as f1:
            for line in f1: data_list.append(json.loads(line))

    corpus_inv_dict = {}
    with open(dataset_path+"/corpus.jsonl", "r") as f4:
        for line in f4:
            corpus_line = json.loads(line)
            corpus_inv_dict[corpus_line["text"]] = corpus_line["_id"]
    
    parsed_data = []
    with open(dataset_path+"/queries.jsonl", "a") as f:
        for i, data in enumerate(tqdm(data_list)):
            data_dict = collections.OrderedDict()

            if dataset_name == "ARC-Challenge":
                data_dict["_id"] = "ARC-Challenge-tq-"+str(i)
                data_dict["text"] = data["prompt"]
                document = corpus_inv_dict[data["choices"][data["answer_idx"]]]
            elif dataset_name == "hellaswag":
                data_dict["_id"] = "hellaswag-tq-"+str(i)
                data_dict["text"] = data["ctx"]
                document = corpus_inv_dict[data["endings"][data["label"]]]
            elif dataset_name == "quail":
                data_dict["_id"] = "quail-tq-"+str(i)
                data_dict["text"] = "Context: " + data["context"] + " Question: " + data["question"]
                document = corpus_inv_dict[data["answers"][int(data["correct_answer_id"])]]
            elif dataset_name == "TempReason-l1":
                data_dict["_id"] = "TR1-tq-"+str(i)
                data_dict["text"] = data["question"]
                document = corpus_inv_dict[data["text_answers"]["text"][0]]
            elif dataset_name == "winogrande":
                data_dict["_id"] = "wino-tq-"+str(i)
                data_dict["text"] = "Sentence: " + data["sentence"]
                document = corpus_inv_dict[data["option1"] if int(data["answer"]) == 1 else data["option2"]]

            data_dict["metadata"] = {}
            f.write( json.dumps(data_dict, sort_keys=False) + "\n" )
            parsed_data.append((data_dict, document))

    with open(dataset_path+"/qrels/train.tsv", "w") as f3:
        tw = csv.writer(f3, delimiter='\t')
        tw.writerow(['query-id', 'corpus-id', 'score'])
        for data_dict, document in parsed_data:
            tw.writerow([data_dict["_id"], document, 1])
    
    print(f"Parsed {dataset_name}: {str(len(parsed_data))} training examples")

def clean_directory(dataset_name, download_path):
    dataset_path = os.path.join(download_path, dataset_name)
    files_to_keep = {"queries.jsonl", "corpus.jsonl", "insruction.txt", dataset_name+".py"}
    folder_to_keep = "qrels"

    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)

        if os.path.isfile(item_path):
            if item not in files_to_keep:
                os.remove(item_path)

        elif os.path.isdir(item_path):
            if item != folder_to_keep:
                shutil.rmtree(item_path)

if __name__=="__main__":
    download_path = args.download_path
    os.makedirs(download_path, exist_ok=True)

    for dataset_name in rarb_datasets_with_labels:
        if os.path.exists(os.path.join(download_path, dataset_name)):
            print(f"{dataset_name} directory already exists!\n")
            continue
        
        download_and_unzip(dataset_name, download_path)
        parse_dataset_with_labels(dataset_name, download_path)
        clean_directory(dataset_name, download_path)
        print()

    for dataset_name in rarb_datasets_without_labels:
        if os.path.exists(os.path.join(download_path, dataset_name)):
            print(f"{dataset_name} directory already exists!\n")
            continue
        
        download_and_unzip(dataset_name, download_path)
        parse_dataset_without_labels(dataset_name, download_path)
        clean_directory(dataset_name, download_path)
        print()

    print("Download finished!")

    