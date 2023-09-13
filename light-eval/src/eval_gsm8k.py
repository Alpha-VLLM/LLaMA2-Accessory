import re
import argparse
import json
import jsonlines
from fractions import Fraction
import re
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

def get_args_parser():
    parser = argparse.ArgumentParser('light-eval', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/gsm8k/')
    parser.add_argument('--batch_size', type=int, default=8)
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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_ans(completion):

    text = re.split("Question:", completion, flags=re.IGNORECASE)[0]
    text = re.split("answer is ", text, flags=re.IGNORECASE)
    
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
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

    while len(tokenizer.encode(prompt, bos=True, eos=False)) + 1> model_max_context - 512: # bos token
        prompt_split = prompt.split("\n\n")
        prompt_split.pop(1)
        prompt = '\n\n'.join(prompt_split)

    return prompt

def run_infer(model, max_seq_len, data_path, infer_path, overwrite = False):

    infer_file = os.path.join(infer_path, f'gsm8k_infer.jsonl')
    if not overwrite and os.path.exists(infer_path):
        print(f"{infer_file} existed, skip!")
        return
    
    test_set = []
    answer_set = []
    fewshot_prompt = open("prompt/gsm8k_prompt.txt").read()
    with open(os.path.join(data_path, 'gsm8k_test.jsonl'),"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            full_prompt = resize_prompt(
                model.tokenizer,
                max_seq_len,
                fewshot_prompt + 
                "\n\nQuestion: " + 
                item["question"] + 
                "\nAnswer: Let's think step by step.\n" 
            )
            test_set.append(full_prompt)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            answer_set.append(temp_ans)

    batch_test_set = batch_data(test_set, batch_size=8)

    res_completions = []
    for batch_input in tqdm(batch_test_set, position=0, leave=True):

        outputs = model.generate(prompts=batch_input, images=None, max_gen_len=512)

        for output in outputs:
            res_completions.append(output)
    
    with jsonlines.open(infer_file, mode='w') as writer:
        for (completion, prompt_answer) in zip(res_completions, answer_set):
            record = {
            'completion': completion,
            'target_ans': prompt_answer
            }
            writer.write(record)

def run_eval(infer_path):

    infer_file = os.path.join(infer_path, 'gsm8k_infer.jsonl')
    assert os.path.exists(infer_file) , f'ERROR: please run inference first!' 

    score = {}
    results = []
    invalid_outputs = []
    with jsonlines.open(infer_file) as f:
        for item in f.iter(type=dict, skip_invalid=True):
            pred = extract_ans(item['completion'])
            if pred != None:
                results.append(float(pred) == float(item['target_ans']))
            else:
                results.append(False)
                temp = {
                    'output_split': re.split('Quetion:', item['completion'], flags=re.IGNORECASE)[0], 
                    'answer': item['target_ans'] 
                }
                invalid_outputs.append(temp)

    score['TOTAL_AVERAGE'] = '%.4f' %(sum(results) / len(results))

    return score, invalid_outputs 

def main(args):

    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    infer_path = os.path.join('results', model_name, 'gsm8k/infer')
    os.makedirs(infer_path, exist_ok=True)
    eval_path = os.path.join('results', model_name, 'gsm8k/eval')
    os.makedirs(eval_path, exist_ok=True)

    model = load(args)
    
    run_infer(model, args.max_seq_len, args.data_dir, infer_path, args.overwrite)
    score, invalid_outputs = run_eval(infer_path)

    with open(os.path.join(eval_path, 'run_results.json'), 'w') as f:
        json.dump(score, f, ensure_ascii=False, indent=2) 

    with open(os.path.join(eval_path, 'debug_invalid_outputs.jsonl'), 'w') as outfile:
        for entry in invalid_outputs:
            json.dump(entry, outfile, ensure_ascii=False,indent=2)
            outfile.write('\n')

if __name__ == "__main__":

    args = get_args_parser().parse_args()
    main(args)
