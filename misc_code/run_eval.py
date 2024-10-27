import sys
import mteb
import argparse

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model_name_or_path', default='intfloat/e5-small-v2', type=str, metavar='N', help='which model to use')
parser.add_argument('--task', default='QuoraRetrieval', type=str, help='which task to use')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--max_length', default=4096, type=int, help='maximum text length')
parser.add_argument('--n_ic_examples', default=5, type=int, help='number of in-context examples')
parser.add_argument('--random_ic_prompt', default=False, type=bool, help='fixed ic prompt')
parser.add_argument('--doc_only', default=False, type=bool, help='doc only')
parser.add_argument('--save_predictions', default=False, type=bool, help='save predictions')
parser.add_argument('--filter', default=False, type=bool, help='save predictions')

args = parser.parse_args()

model_name = args.model_name_or_path
batch_size = args.batch_size
max_length = args.max_length
n_ic_examples = args.n_ic_examples
task = args.task

model = mteb.get_model(model_name)

if "dupstack" in task.lower():
    tasks = ["CQADupstackAndroidRetrieval", "CQADupstackEnglishRetrieval", "CQADupstackGamingRetrieval", "CQADupstackGisRetrieval", "CQADupstackMathematicaRetrieval", "CQADupstackPhysicsRetrieval", "CQADupstackProgrammersRetrieval", "CQADupstackStatsRetrieval", "CQADupstackTexRetrieval", "CQADupstackUnixRetrieval", "CQADupstackWebmastersRetrieval", "CQADupstackWordpressRetrieval"]
else:
    tasks = [task]

evaluation = mteb.MTEB(tasks=tasks, n_ic_examples=n_ic_examples, random_ic_prompt=args.random_ic_prompt, doc_only=args.doc_only)

if "mistral-7b-instruct" in model_name.lower() or "sfr" in model_name.lower():
    print(f"Running model of type mistral-7b-instruct")
    results = evaluation.run(model, output_folder=f"results_pred_{args.n_ic_examples}ic_{args.random_ic_prompt}random/{model_name}", eval_splits=["test"], encode_kwargs={"batch_size": batch_size, "max_length": max_length}, save_predictions=args.save_predictions)
elif "llm2vec" in model_name.lower() or "sheared" in model_name.lower() or "llama" in model_name.lower() or "qwen" in model_name.lower():
    results = evaluation.run(model, output_folder=f"results_pred_{args.n_ic_examples}ic_{args.random_ic_prompt}random/{model_name}", eval_splits=["test"], encode_kwargs={"batch_size": batch_size}, save_predictions=args.save_predictions)
else:
    results = evaluation.run(model, output_folder=f"results_pred_{args.n_ic_examples}ic_{args.random_ic_prompt}random/{model_name}", eval_splits=["test"], encode_kwargs={"batch_size": batch_size, "max_length": max_length}, save_predictions=args.save_predictions)