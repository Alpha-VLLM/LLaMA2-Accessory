
import argparse
# import re
import numpy as np
import torch
import pandas as pd
# from thefuzz import process
from tqdm import tqdm
import json

import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

from eval_utils.ceval_categories import TASK_NAME_MAPPING, hard_list 

choices = ["A", "B", "C", "D"]

def get_args_parser():
    parser = argparse.ArgumentParser('ligit-eval', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/ceval')
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
                        help='directory containing pretrained checkpoints')
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

def format_example(line, include_answer=True):
    example = line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'
    if include_answer:
        example += '\n答案：' + line["answer"] + '\n\n'
    else:
        example += '\n答案：'
    return example

def generate_few_shot_prompt(subject, dev_df, ntrain=-1):
    prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
    if ntrain == -1:
        ntrain = dev_df.shape[0]
    for i in range(ntrain):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True
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
    for task in TASK_NAME_MAPPING.keys():

        print('Testing %s ...' % task)
        val_file_path = os.path.join(data_path, "val", f"{task}_val.csv")
        val_df = pd.read_csv(val_file_path)
        dev_file_path = os.path.join(data_path, "dev", f"{task}_dev.csv")
        dev_df = pd.read_csv(dev_file_path)
        few_shot_prompt = generate_few_shot_prompt(task, dev_df, ntrain) if few_shot else []

        results = []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
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

            results.append(pred == row['answer'])
        total_results[task] = sum(results) / len(results)

    return total_results
            
def cal_ceval(res):
    results = {}
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0
    hard_cnt = 0
    hard_acc_sum = 0.0
    for tt in res.keys():
        name = tt.split("-")[-1]
        acc_sum += float(res[tt])
        cnt += 1
        class_ = TASK_NAME_MAPPING[name][2]
        if class_ not in acc_sum_dict:
            acc_sum_dict[class_] = 0.0
            acc_norm_sum_dict[class_] = 0.0
            cnt_dict[class_] = 0.0
        if name in hard_list:
            hard_cnt += 1
            hard_acc_sum += float(res[tt])
        acc_sum_dict[class_] += float(res[tt])
        cnt_dict[class_] += 1
    print("\n\n\n")
    for k in ["STEM", "Social Science", "Humanities", "Other"]:
        if k in cnt_dict:
            results[k] = "%.4f" %(acc_sum_dict[k] / cnt_dict[k])
            print("%s acc: %.4f " % (k, acc_sum_dict[k] / cnt_dict[k]))
    if hard_cnt > 0:
        results['Hard'] = "%.4f" %(hard_acc_sum / hard_cnt)
        print("Hard acc:%.4f " % (hard_acc_sum / hard_cnt))
    results ['AVERAGE'] = "%.4f" % (acc_sum / cnt)
    print("AVERAGE acc:%.4f " % (acc_sum / cnt))
    return results

def main(args):

    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    eval_path = os.path.join('results', model_name, 'ceval/eval')
    os.makedirs(eval_path, exist_ok=True)
    
    model = load(args)

    result_path = os.path.join(eval_path, 'run_results.json')
    if not args.overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        return

    subjects_result = run_infer_eval(model, args.max_seq_len, args.data_dir, args.ntrain)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:

        score = cal_ceval(subjects_result)
        with open(result_path, 'w') as f:
            json.dump(score, f, ensure_ascii=False, indent=2) 

if __name__ == "__main__":

    args = get_args_parser().parse_args()
    main(args)
