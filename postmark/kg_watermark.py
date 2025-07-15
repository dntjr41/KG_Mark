import os
import tqdm
import torch
import json
import argparse
from models import KGWatermarker

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["opengen", "factscore", "lfqa"], default="opengen")
parser.add_argument('--output_path', type=str, default="/home/wooseok/KG_Mark/outputs/test/test_subgraph.jsonl")
parser.add_argument('--llm', type=str, choices=["gpt-4", "llama-3-8b", "llama-3-8b-chat", "mistral-7b-inst"], default="llama-3-8b-chat")
parser.add_argument('--embedder', type=str, choices=["openai", "nomic"], default="openai")
parser.add_argument('--inserter', type=str, choices=["gpt-4o", "llama-3-70b-chat"], default="llama-3-70b-chat")
parser.add_argument('--ratio', type=float, default=0.12)
parser.add_argument('--iterate', type=str, default="v2")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--n', type=int)
args = parser.parse_args()
print(args)

# Set CUDA_VISIBLE_DEVICES to force using only the specified GPU
if args.device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    print(f"Set CUDA_VISIBLE_DEVICES to {args.device_id}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")

if args.dataset == "opengen":
    input_path = "data/opengen_500.jsonl"
elif args.dataset == "lfqa":
    input_path = "data/lfqa.jsonl"
elif args.dataset == "factscore":
    input_path = "data/factscore.jsonl"
else:
    raise NotImplementedError

output_path = args.output_path

watermarker = KGWatermarker(args.llm,
                            args.embedder,
                            args.inserter,
                            ratio=args.ratio,
                            iterate=args.iterate,
                            device_id=args.device_id)

if args.n:
    input_data = [json.loads(line) for line in open(input_path, 'r')][:args.n]
else:
    input_data = [json.loads(line) for line in open(input_path, 'r')]
print(f"Loaded {len(input_data)} examples for {args.dataset}.")

def append_to_output_file(output_path, generation_record):
    with open(output_path, 'a') as fout:
        fout.write(json.dumps(generation_record) + "\n")

for idx, dd in tqdm.tqdm(enumerate(input_data)):
    prefix = dd["prefix"]
    target = ", ".join(dd["targets"])
    # delta 값은 필요에 따라 조정 (기본 0.3)
    result = watermarker.insert_watermark(prefix, target, max_tokens=1500, delta=0.3)
    append_to_output_file(output_path, result)