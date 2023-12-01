import argparse
import jsonlines
import json
import time
import pandas as pd
from tqdm import tqdm
import torch

import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

from eval_utils.mmlu_categories import TASKS

choices = ["A", "B", "C", "D"]

def get_args_parser():
    parser = argparse.ArgumentParser('light-eval', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/mmlu/')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--overwrite', action="store_true", default=False, help="Overwrite existed results")
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

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def generate_few_shot_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
    
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

def extract_ans(ans):
    ans = ans.strip()
    if ans != '':
        return ans[-1]
    else:
        return None

def batch_data(prompts, batch_size=1):
    batch_data = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_size:
            batch_data.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_data.append(mini_batch)
    return batch_data 

def resize_prompt(tokenizer, model_max_context, prompt):

    while len(tokenizer.encode(prompt, bos=True, eos=False)) + 1> model_max_context - 1024: # bos token
        prompt_split = prompt.split("\n\n")
        prompt_split.pop(1)
        prompt = '\n\n'.join(prompt_split)

    return prompt

def run_infer(model, max_seq_len, tasks, infer_path, ntrain=5, overwrite = False):

    for task in tasks:

        task_infer_path = os.path.join(infer_path, f'{task}_infer.jsonl')
        if not overwrite and os.path.exists(task_infer_path):
            print(f"{task_infer_path} existed, skip!")
            continue

        print('Testing %s ...' % task)
        dev_df = pd.read_csv(os.path.join(args.data_dir, "data/dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "data/test", task + "_test.csv"), header=None) 
        few_shot_prompt = generate_few_shot_prompt(dev_df, task, ntrain)

        test_set = []
        answer_set = []
        for i in range(test_df.shape[0]):
            prompt = format_example(test_df, i, include_answer=False)
            full_prompt = resize_prompt(
                model.tokenizer,
                max_seq_len,
                few_shot_prompt+prompt
            )
            
            test_set.append(full_prompt)
            target_ans = test_df.iloc[i, test_df.shape[1]-1]
            answer_set.append(target_ans)
            
        batch_prompt = batch_data(test_set, batch_size=8)

        res_completions = []
        for batch_input in tqdm(batch_prompt):
            
            outputs = model.generate(prompts=batch_input, images=None, max_gen_len=1)
            
            for output in outputs:
                res_completions.append(output)
        
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            
            with jsonlines.open(task_infer_path, mode='w') as writer:
                for (completion, prompt_answer) in zip(res_completions, answer_set):
                    record = {
                    'completion': completion,
                    'target_ans': prompt_answer
                    }
                    writer.write(record)
            writer.close()

def run_eval(tasks, infer_path):
    
    score = {}
    total_results=[]
    invalid_outputs = []
    for task in tasks:

        task_infer_path = os.path.join(infer_path, f'{task}_infer.jsonl')
        assert os.path.exists(task_infer_path) , f'ERROR: please run {task} inference first!' 

        results = []
        invalid = []
        with jsonlines.open(task_infer_path) as f:
            for item in f.iter(type=dict, skip_invalid=True):
                pred = extract_ans(item['completion'])
                if pred == None:
                    invalid.append(
                        {'output': item['completion'],
                        'answer': item['target_ans']}
                    )
                results.append(pred == item['target_ans'])
        score[task] = '%.4f' %(sum(results) / len(results))
        total_results.extend(results)
        invalid_outputs.extend(invalid)
    score['TOTAL_AVERAGE'] = '%.4f' %(sum(total_results) / len(total_results))

    return score, total_results, invalid_outputs

def main(args):
    
    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    infer_path = os.path.join('results', model_name, 'mmlu/infer')
    os.makedirs(infer_path, exist_ok=True)
    eval_path = os.path.join('results', model_name, 'mmlu/eval')
    os.makedirs(eval_path, exist_ok=True)

    model = load(args)

    run_infer(model, args.max_seq_len, TASKS, infer_path, args.ntrain, args.overwrite)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:

        score, _ , invalid_outputs= run_eval(TASKS, infer_path)

        with open(os.path.join(eval_path, 'run_results.json'), 'w') as f:
            json.dump(score, f, ensure_ascii=False, indent=2) 

        with open(os.path.join(eval_path, 'debug_invalid_outputs.jsonl'), 'w') as outfile:
            for entry in invalid_outputs:
                json.dump(entry, outfile, ensure_ascii=False,indent=2)
                outfile.write('\n')


if __name__ == "__main__":

    args = get_args_parser().parse_args()
    main(args)

