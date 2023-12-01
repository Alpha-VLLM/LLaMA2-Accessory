import argparse

import torch
from tqdm import tqdm
import json
import  jsonlines

import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

from human_eval.data import write_jsonl, read_problems
from eval_utils.humaneval_evaluation import evaluate_functional_correctness

def get_args_parser():
    parser = argparse.ArgumentParser('light-eval', add_help=False)
    # Dataset parameters
    parser.add_argument("--overwrite", action="store_true", default=False, 
                        help="Overwrite existed results")
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pretrained checkpoints')
    parser.add_argument('--pretrained_type', type=str, default="consolidated", choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')
    # Parrallel parameters
    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

# load model and tokenizer
def load(args):

    # define the model
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)

    if args.quant:
        print("Quantizing model to 4bit!")

        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)
        
    #print("Model = %s" % str(model))
    model.bfloat16().cuda()
    return model

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout
    )

    return results

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def split_batch(samples: list[str], size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches

@torch.inference_mode()
def generate_batch_completion(
    model: MetaModel, prompt, batch_size
) -> list[str]:
    batch_input = [prompt for _ in range(batch_size)]

    batch_completions = model.generate(
        prompts=batch_input, 
        images=None, 
        max_gen_len=1024, 
        temperature=0.2,
        top_p=0.95)

    return [filter_code(fix_indents(completion)) for completion in batch_completions]

def run_infer(
    model:MetaModel,
    infer_file: str, 
    overwrite: bool = False, 
    num_samples_per_task: int = 10, 
    format_tabs: bool = True 
):

    if not overwrite and os.path.exists(infer_file):
        print(f"{infer_file} existed, skip!")
        return
    
    problems = read_problems()
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:

        with jsonlines.open(infer_file, mode='w') as writer:
            for x in samples:
                writer.write(x)

def main(args):

    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    infer_path = os.path.join('results', model_name, 'humaneval/infer')
    os.makedirs(infer_path, exist_ok=True)
    eval_path = os.path.join('results', model_name, 'humaneval/eval')
    os.makedirs(eval_path, exist_ok=True)

    model = load(args)

    infer_file = os.path.join(infer_path, 'human_infer.jsonl')
    run_infer(model, infer_file, args.overwrite,)
    
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:

        print("Evaluating...")
        score = entry_point(sample_file=infer_file)

        with open(os.path.join(eval_path, 'run_results.json'), 'w') as f:
            json.dump(score, f, ensure_ascii=False, indent=2) 
        print(score)


if __name__ == "__main__":

    args = get_args_parser().parse_args()
    main(args)

