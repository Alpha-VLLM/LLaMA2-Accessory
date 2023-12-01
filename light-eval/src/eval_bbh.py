import argparse
import json
import jsonlines
import re
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

MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]

FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]

def get_args_parser():
        
    parser = argparse.ArgumentParser('light-eval', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/BIG-Bench-Hard')
    parser.add_argument('--ntrain', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--task', type=str, default='all', choices=['all', 'multiple_choice', 'free_form'])
    parser.add_argument('--overwrite', action="store_true", default=False, help="Overwrite existed results")
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
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
    parser.add_argument('--quant', action="store_true", default=False,
                        help="enable quantization to speedup and save memory")
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

def extract_ans(ans, mode):
    ans_line = re.split("Q:", ans, flags=re.IGNORECASE)[0]
    ans_line = re.split('answer is ', ans_line, flags=re.IGNORECASE)
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        match = re.search(r'\(([A-Z])\)*', ans)
        if match:
            return match.group(1)
        match = re.search(r'([A-Z])', ans)
        if match:
            return match.group(1)
        return ans
    elif mode == 'free_form':
        ans = re.split(r'[.#]', ans)[0]
        return ans

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

def run_infer(model, max_seq_len, tasks, data_path, infer_path, mode, overwrite = False):

    for task in tasks:

        task_infer_path = os.path.join(infer_path, f'{task}_infer.jsonl')
        if not overwrite and os.path.exists(task_infer_path):
            print(f"{task_infer_path} existed, skip!")
            continue

        print(f'Testing {task} ...')
        task_data = json.load(open(os.path.join(data_path, f'bbh/{task}.json')))
        with open(os.path.join(data_path, f'cot-prompts/{task}.txt'), 'r') as f:
            task_prompt = f.readlines()[2:]
            task_prompt = ''.join(task_prompt)

        test_set = []
        answer_set = []
        for item in task_data['examples']:
            full_prompt = resize_prompt(
                model.tokenizer,
                max_seq_len,
                task_prompt + 
                '\n\nQ: ' + 
                item['input'] + 
                "\nA: Let's think step by step."
            )
            
            test_set.append(full_prompt)
            if mode == 'multiple_choice':
                answer_set.append(item['target'][1])
            elif mode == 'free_form':
                answer_set.append(item['target'])
            
        batch_prompt = batch_data(test_set, batch_size=8)
        res_completions = []
        for batch_input in tqdm(batch_prompt):
            
            outputs = model.generate(prompts=batch_input, images=None, max_gen_len=1024)
            
            for output in outputs:
                res_completions.append(output)

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            with jsonlines.open(task_infer_path, mode='w') as writer:
                for (prompt, completion, prompt_answer) in zip(task_data['examples'], res_completions, answer_set):
                    record = {'prompt': prompt,
                        'completion': completion,
                        'target_ans': prompt_answer
                    }
                    writer.write(record)

def run_eval(tasks, infer_path, mode):
    
    score = {}
    total_results=[]
    for task in tasks:

        task_infer_path = os.path.join(infer_path, f'{task}_infer.jsonl')
        assert os.path.exists(task_infer_path) , f'ERROR: please run {task} inference first!' 

        results = []
        with jsonlines.open(task_infer_path) as f:
            for item in f.iter(type=dict, skip_invalid=True):
                pred = extract_ans(item['completion'], mode)
                results.append(pred == item['target_ans'])
        score[task] = '%.4f' %(sum(results) / len(results))
        total_results.extend(results)
    score['TOTAL_AVERAGE'] = '%.4f' %(sum(total_results) / len(total_results))

    return score, total_results
                

def main(args, multiple_choice_tasks=MULTIPLE_CHOICE_TASKS, free_form_tasks=FREE_FORM_TASKS):

    run_multiple_choice = args.task == 'all' or args.task == 'multiple_choice'
    run_free_form = args.task == 'all' or args.task == 'free_form'

    path_split = args.pretrained_path[0].split('/') if isinstance(args.pretrained_path,list) else args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    infer_path = os.path.join('results', model_name, 'bbh/infer')
    os.makedirs(infer_path, exist_ok=True)
    eval_path = os.path.join('results', model_name, 'bbh/eval')
    os.makedirs(eval_path, exist_ok=True)

    model = load(args)
    

    if run_multiple_choice:
        run_infer(model, args.max_seq_len, multiple_choice_tasks, args.data_dir, infer_path, 'multiple_choice', args.overwrite)
    if run_free_form:
        run_infer(model, args.max_seq_len, free_form_tasks, args.data_dir, infer_path, 'free_form', args.overwrite)
    
    
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:

        score = {}
        total_results = []

        if run_multiple_choice:    
            score['multiple_choice'], task_results = run_eval(multiple_choice_tasks, infer_path, mode='multiple_choice')
        total_results.extend(task_results)

        if run_free_form:
            score['free_form'], task_results = run_eval(free_form_tasks, infer_path, mode='free_form')
        total_results.extend(task_results)

        if args.task == 'all':
            score['TOTAL'] = '%.4f' %(sum(total_results) / len(total_results))
        
        result_path = os.path.join(eval_path, 'run_results.json')

        with open(result_path, 'w') as f:
            json.dump(score, f, ensure_ascii=False, indent=2) 
    
    return 

if __name__ == '__main__':

    args = get_args_parser().parse_args()
    main(args)
