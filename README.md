RARe - Retrieval Augmented Retrieval With In-Context Examples
==================================================

## 1. Overview

## 2. Setup
### 2.2 Installation
```bash
sh setup.sh
```

### 2.3 Preprocessing

Downloading supervised training data from the [Echo Embeddings Repository](https://github.com/jakespringer/echo-embeddings) for training from retriever checkpoint experiment.
```bash
cd data
wget "https://drive.usercontent.google.com/download?id=1YqgaJIzmBIH37XBxpRPCVzV_CLh6aOI4&export=download"
tar -xvf echo-data.tar
rm -r echo-data.tar
cd ../
```

Preprocessing RAR-b benchmark for evaluation
```bash
cd misc_code
python process_rarb.py
cd ../
```

## 3. Running Experiments

### 3.1 Training

#### 3.1.1 Training from Retreiver Checkpoint
Configuration files are provided in ```llm2vec/train_configs/supervised```. We mainly use ```E5-Instruct.json``` and ```MetaLlama3-Supervised.json```.

```bash
cd LLM2Vec
sh run.sh
```
Trains [E5-Mistral-7B-Instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) with 5 in-context examples using retrieved examples with BM25.

#### 3.1.2 Training from LLM (Decoder-Only) Checkpoint

```bash
cd tevatron
sh run.sh
```
Trains [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) with 5 in-context examples using retrieved examples with BM25.

### 3.2 Evaluation

The ```run_eval.sh``` script in the ```misc_code/``` folder provides an example of running evaluation with 5 in-context examples.

```bash
cd misc_code
sh run_eval.sh
```

You may need to modify ```e5_models.py``` ```llm2vec_models.py``` and ```repllama_models.py``` in ```mteb/models``` to include the paths to newly trained models. Examples are provided in each of these files.

##### Some of the code was forked from the following repositories
* [llm2vec](https://github.com/McGill-NLP/llm2vec)
* [tevatron](https://github.com/texttron/tevatron)
* [mteb](https://github.com/embeddings-benchmark/mteb)

## Cite

If our work was helpful in your research, please kindly cite us as follows:
```

```