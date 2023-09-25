import torch
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import json

import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc

from fairscale.nn.model_parallel import initialize as fs_init

from eval_utils.cmmlu_categories import name_en2zh, subcategories, categories

TASK_NAME_MAPPING = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                TASK_NAME_MAPPING[k].append(subject)

choices = ["A", "B", "C", "D"]

def get_args_parser():
        
    parser = argparse.ArgumentParser('ligit-eval', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/cmmlu')
    parser.add_argument('--ntrain', type=int, default=5)
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
                        help='directory containing pre-trained checkpoints')
    parser.add_argument('--pretrained_type', type=str, default="consolidated", choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')
    parser.add_argument('--max_seq_len', default=2048, type=int,
                        help='max input sequence length, which should be adjusted accordingly to the model')
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
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
    print(f"load pretrained from {args.pretrained_path}")
    misc.load_pretrained(args.pretrained_path, args.pretrained_type, model)
    print("Model = %s" % str(model))
    model.bfloat16().cuda()

    return model

def format_example(line, include_answer=True):
    example = "问题：" + line["Question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\n答案：" + line["Answer"] + "\n\n"
    else:
        example += "\n答案："
    return example

def generate_few_shot_prompt(task, dev_df, ntrain=-1):
    prompt = "以下是关于{}的单项选择题，请直接给出正确答案的选项。\n\n".format(name_en2zh[task])
    # prompt = ""
    if ntrain == -1:
        ntrain = dev_df.shape[0]
    for i in range(ntrain):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt

def extract_ans_by_logits(tokenizer, logits):

    assert logits.shape[0] == 1
    logits = logits.flatten()
    
    softval = torch.nn.functional.softmax(
        torch.tensor(
            [
                logits[tokenizer.encode(
                    "A", bos=False, eos=False)[0]],
                logits[tokenizer.encode(
                    "B", bos=False, eos=False)[0]],
                logits[tokenizer.encode(
                    "C", bos=False, eos=False)[0]],
                logits[tokenizer.encode(
                    "D", bos=False, eos=False)[0]],
            ]
        ),
        dim=0,
    )
    if softval.dtype in {torch.bfloat16, torch.float16}:
        softval = softval.to(dtype=torch.float32)
    probs = softval.detach().cpu().numpy()

    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    return pred

def resize_prompt(tokenizer, model_max_context, prompt):

    while len(tokenizer.encode(prompt, bos=True, eos=True)) > model_max_context - 1: # bos token
        prompt_split = prompt.split("\n\n")
        prompt_split.pop(1)
        prompt = '\n\n'.join(prompt_split)

    return prompt

def run_infer_eval(model, max_seq_len, data_path, ntrain=-1, few_shot = True):

    total_results = {}
    for task in subcategories.keys():

        print('Testing %s ...' % task)
        test_file_path = os.path.join(data_path, "test", f"{task}.csv")
        test_df = pd.read_csv(test_file_path)
        dev_file_path = os.path.join(data_path, "dev", f"{task}.csv")
        dev_df = pd.read_csv(dev_file_path)
        few_shot_prompt = generate_few_shot_prompt(task, dev_df, ntrain) if few_shot else []

        results = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            prompt = format_example(row, include_answer=False)
            full_prompt = resize_prompt(
                model.tokenizer,
                max_seq_len,
                few_shot_prompt+prompt
            )
            output = model.generate(
                prompts=[full_prompt], 
                images=None,
                max_gen_len=100,
                return_logits=True
            )
            pred = extract_ans_by_logits(tokenizer = model.tokenizer, logits=output)

            results.append(pred == row['Answer'])
        total_results[task] = sum(results) / len(results)

    return total_results

def cal_cmmlu(res):
    print("\n\n\n")
    results = {}
    res = {k.split("-")[-1]: float(v) for k, v in res.items()}
    for k, v in TASK_NAME_MAPPING.items():
        avg_acc = np.mean(list(map(lambda x: res[x], v)))
        print("%s acc: %.4f " % (k, avg_acc))
        results[k] = "%.4f" % avg_acc
    avg_all_acc = np.mean(list(res.values()))
    print("AVERAGE acc:%.4f " % avg_all_acc)
    results ['AVERAGE'] = "%.4f" % avg_all_acc
    
    return results

def main(args):

    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    eval_path = os.path.join('results', model_name, 'cmmlu/eval')
    os.makedirs(eval_path, exist_ok=True)
    
    model = load(args)

    result_path = os.path.join(eval_path, 'run_results.json')
    if not args.overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        return

    subjects_result = run_infer_eval(model, args.max_seq_len, args.data_dir, args.ntrain)
    score = cal_cmmlu(subjects_result)

    if torch.distributed.get_rank() == 0:
        torch.distributed.barrier()
        with open(result_path, 'w') as f:
            json.dump(score, f, ensure_ascii=False, indent=2) 

if __name__ == "__main__":
    
    args = get_args_parser().parse_args()
    main(args)
