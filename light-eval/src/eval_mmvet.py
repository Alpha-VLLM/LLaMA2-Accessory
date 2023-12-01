import sys
import os
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel

import argparse
import torch
import openai
import pandas as pd
import numpy as np
from collections import Counter
import time
from PIL import Image

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from data.alpaca import transform_val
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pretrained checkpoints')

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
                        help="enable quantization")
    parser.add_argument("--model_name", type=str, default="llama_accessory_2")


    parser.add_argument("--max_gen_len", type=int, default=516)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    
    
    parser.add_argument("--openai_key", type=str, default= "sk-xxxxxxxxxxxx")
    parser.add_argument("--image_folder", type=str, default="/data1/zyc/LLaMA2-Accessory/accessory/accessory_mm_eval/MM-vet/mm-vet/images")
    parser.add_argument("--question_file", type=str, default="/data1/zyc/LLaMA2-Accessory/accessory/accessory_mm_eval/MM-vet/mm-vet/mm-vet.json")
    parser.add_argument("--answers_file", type=str, default="mmvet-answers.jsonl")
    parser.add_argument("--use_sub_set", choices=["True","False"], default="False")
    parser.add_argument("--mode", choices=["all","eval"], default="all")
    return parser



def format_prompt(prompt):
    prompt_t=f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
    return prompt_t

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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
        
    print("Model = %s" % str(model))
    model.bfloat16().cuda()
    return model

@ torch.inference_mode()
def generate_output(model, img_path, prompt):
    print("image path:", img_path)
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        image = transform_val(image).unsqueeze(0)
    else:
        image = None
    _prompt = format_prompt(prompt)

    if image is not None:
        image = image.cuda()
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        results = model.generate([_prompt], image, max_gen_len=512, temperature=0.1, top_p=0.7)
    text_output = results[0].strip()
    return text_output

def eval_MMVet_benchmark(model, args):

    answers_file = os.path.expanduser(args.answers_file)
    results={}
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    with open(args.question_file, "r") as file:
        data = json.loads(file.read())
    for key, value in data.items():
        idx = key
        image_file = value["imagename"]
        qs=value["question"]

        image_path = os.path.join(args.image_folder, image_file)
        prompt = qs
        times=0
        output_text = generate_output(
            model = model,
            img_path = image_path,
            prompt = prompt)

        res= {
            "idx": idx,
            "output_text": output_text
        }

        key = idx
        context= output_text
        results[key] = context        

    output_json = json.dumps(results, indent=4)
    ans_file.write(output_json)
    ans_file.flush()
    ans_file.close()
    

    
def scores(args):
    gpt_model = "gpt-4-0613"
    prompt = open("prompt/MMVet_prompt.txt").read()

    mmvet_path="data/MM-Vet/"
    use_sub_set = args.use_sub_set
    decimal_places = 1 # number of decimal places to round to
    if use_sub_set:
        bard_set_file = os.path.join(mmvet_path, "bard_set.json")
        with open(bard_set_file, 'r') as f:
            sub_set = json.load(f)
        sub_set_name = 'bardset'
        sub_set_name = sub_set_name + '_'
    else:
        sub_set = None
        sub_set_name = ''

    mmvet_metadata = os.path.join(mmvet_path, "mm-vet.json")

    with open(mmvet_metadata, 'r') as f:
        data = json.load(f)


    counter = Counter()
    cap_set_list = []
    cap_set_counter = []
    len_data = 0
    for id, value in data.items():
        if sub_set is not None and id not in sub_set:
            continue
        question = value["question"]
        answer = value["answer"]
        cap = value["capability"]
        cap = set(cap)
        counter.update(cap)
        if cap not in cap_set_list:
            cap_set_list.append(cap)
            cap_set_counter.append(1)
        else:
            cap_set_counter[cap_set_list.index(cap)] += 1
        
        len_data += 1

    sorted_list = counter.most_common()
    columns = [k for k, v in sorted_list]
    columns.append("total")
    columns.append("std")
    columns.append('runs')
    df = pd.DataFrame(columns=columns)


    cap_set_sorted_indices = np.argsort(-np.array(cap_set_counter))
    new_cap_set_list = []
    new_cap_set_counter = []
    for index in cap_set_sorted_indices:
        new_cap_set_list.append(cap_set_list[index])
        new_cap_set_counter.append(cap_set_counter[index])

    cap_set_list = new_cap_set_list
    cap_set_counter = new_cap_set_counter
    cap_set_names = ["_".join(list(cap_set)) for cap_set in cap_set_list]

    columns2 = cap_set_names
    columns2.append("total")
    columns2.append("std")
    columns2.append('runs')
    df2 = pd.DataFrame(columns=columns2)
    
    model = args.model_name
    result_path = f"MMVet_result/{args.model_name}"

    num_run = 1 # we set it as 5 in the paper
    model_results_file = os.path.join(result_path, f"{model}.json")
    # grade results for each samplex to save
    grade_file = f'{model}_{gpt_model}-grade-{num_run}runs.json'
    grade_file = os.path.join(result_path, grade_file)

    # score results regarding capabilities/capability integration to save
    cap_score_file = f'{model}_{sub_set_name}{gpt_model}-cap-score-{num_run}runs.csv'
    cap_score_file = os.path.join(result_path, cap_score_file)
    cap_int_score_file = f'{model}_{sub_set_name}{gpt_model}-cap-int-score-{num_run}runs.csv'
    cap_int_score_file = os.path.join(result_path, cap_int_score_file)
    
    
    with open(model_results_file) as f:
        results = json.load(f)
    if os.path.exists(grade_file):
        with open(grade_file, 'r') as f:
            grade_results = json.load(f)
    else:
        grade_results = {}

    def need_more_runs():
        need_more_runs = False
        if len(grade_results) > 0:
            for k, v in grade_results.items():
                if len(v['score']) < num_run:
                    need_more_runs = True
                    break
        return need_more_runs or len(grade_results) < len_data
    
    while need_more_runs():
        for j in range(num_run):
            print(f'eval run {j}')
            for id, line in tqdm(data.items()):
                if sub_set is not None and id not in sub_set:
                    continue
                if id in grade_results and len(grade_results[id]['score']) >= (j + 1):
                    continue

                model_pred = results[id]
                
                question = prompt + '\n' + ' | '.join([line['question'], line['answer'].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "), model_pred, ""])
                messages = [
                {"role": "user", "content": question},
                ]

                if id not in grade_results:
                    sample_grade = {'model': [], 'content': [], 'score': []}
                else:
                    sample_grade = grade_results[id]

                
                grade_sample_run_complete = False
                temperature = 0.0
                openai.api_key = args.openai_key
                
                while not grade_sample_run_complete:
                    try:
                        response = openai.ChatCompletion.create(
                            model=gpt_model,
                            max_tokens=3,
                            temperature=temperature,
                            messages=messages)
                        content = response['choices'][0]['message']['content']
                        flag = True
                        try_time = 1
                        while flag:
                            try:
                                content = content.split(' ')[0].strip()
                                score = float(content)
                                if score > 1.0 or score < 0.0:
                                    assert False
                                flag = False
                            except:
                                question = prompt + '\n' + ' | '.join([line['question'], line['answer'].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "), model_pred, ""]) + "\nPredict the correctness of the answer (digit): "
                                messages = [
                                {"role": "user", "content": question},
                                ]
                                response = openai.ChatCompletion.create(
                                    model=gpt_model,
                                    max_tokens=3,
                                    temperature=temperature,
                                    messages=messages)
                                content = response['choices'][0]['message']['content']
                                try_time += 1
                                temperature += 0.5
                                print(f"{id} try {try_time} times")
                                print(content)
                                if try_time > 5:
                                    score = 0.0
                                    flag = False
                        grade_sample_run_complete = True
                    except:
                        # gpt4 may have token rate limit
                        print("sleep 30s")
                        time.sleep(30)

                if len(sample_grade['model']) >= j + 1:
                    sample_grade['model'][j] = response['model']
                    sample_grade['content'][j] = content
                    sample_grade['score'][j] = score
                else:
                    sample_grade['model'].append(response['model'])
                    sample_grade['content'].append(content)
                    sample_grade['score'].append(score)
                grade_results[id] = sample_grade

                with open(grade_file, 'w') as f:
                    json.dump(grade_results, f, indent=4)
                    
    assert not need_more_runs()
    cap_socres = {k: [0.0]*num_run for k in columns[:-2]}
    counter['total'] = len_data

    cap_socres2 = {k: [0.0]*num_run for k in columns2[:-2]}
    counter2 = {columns2[i]:cap_set_counter[i] for i in range(len(cap_set_counter))}
    counter2['total'] = len_data

    for k, v in grade_results.items():
        if sub_set is not None and k not in sub_set:
            continue
        for i in range(num_run):
            score = v['score'][i]
            caps = set(data[k]['capability'])
            for c in caps:
                cap_socres[c][i] += score
            
            cap_socres['total'][i] += score

            index = cap_set_list.index(caps)
            cap_socres2[cap_set_names[index]][i] += score
            cap_socres2['total'][i] += score

    for k, v in cap_socres.items():
        cap_socres[k] = np.array(v) / counter[k] *100


    std = round(cap_socres['total'].std(), decimal_places)
    total_copy = cap_socres['total'].copy()
    runs = str(list(np.round(total_copy, decimal_places)))

    for k, v in cap_socres.items():
        cap_socres[k] = round(v.mean(), decimal_places)
 

    cap_socres['std'] = std
    cap_socres['runs'] = runs
    
    # when use subset, please note the column order is different from the full set
    # because it ranks by numbers of capabilties/capability integrations
    
    print("#####result1#####")
    for key, value in cap_socres.items():
        print(f"{key}: {value}")
    
    df.loc[model] = cap_socres


    for k, v in cap_socres2.items():
        cap_socres2[k] = round(np.mean(np.array(v) / counter2[k] *100), decimal_places)
    cap_socres2['std'] = std
    cap_socres2['runs'] = runs
    df2.loc[model] = cap_socres2

    print("#####result2#####")
    for key, value in cap_socres2.items():
        print(f"{key}: {value}")

    df.to_csv(cap_score_file)
    df2.to_csv(cap_int_score_file)
    





args = get_args_parser().parse_args()
model = load(args)
if args.mode == 'all':
    eval_MMVet_benchmark(model, args)
    scores(args)
elif args.mode == 'inference':
    eval_MMVet_benchmark(model, args)
elif args.mode == 'eval':
    scores(args)
else:
    print("please choose from 'eval' and 'all'")
