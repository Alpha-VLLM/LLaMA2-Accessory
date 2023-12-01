import sys
import os
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
import argparse
import torch


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

import openai
import time
NUM_SECONDS_TO_SLEEP = 0.5


def get_args_parser():
    parser = argparse.ArgumentParser('llava_benchmark evaluation', add_help=False)
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
    
    #llava benchmark setting
    parser.add_argument("--image_folder", type=str, default="path/to/images")
    parser.add_argument("--model_name", type=str, default="llama_accessory_2")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_gen_len", type=int, default=516)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--question_file", type=str, default="path/to/questions.jsonl")
    parser.add_argument("--answers_file", type=str, default="yourpath/to_save/answers.jsonl")
    
    # gpt4 settings
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')

    parser.add_argument("--context", type=str, default= "path/to/llava-bench-in-the-wild/context.jsonl")
    parser.add_argument("--answer-list", nargs='+', default=[])
    parser.add_argument("--rule", type=str, default= "path/to/llava/eval/table/rule.json")
    parser.add_argument("--output", type=str, default= "yourpath/to_save/review.jsonl")
    parser.add_argument("--openai_key", type=str, default= "sk-xxxxxxxxxxxx")

    parser.add_argument("--mode", choices=['inference', 'eval', 'show', 'all'], default='all')


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
        
    #print("Model = %s" % str(model))
    model.bfloat16().cuda()
    return model


@ torch.inference_mode()
def generate_output(model, img_path,prompt):
   
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



def eval_llava_benchmark(model, args):
    model_name = args.model_name
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        image_path = os.path.join(args.image_folder, image_file)
        prompt = qs
        times=0
        output_text = generate_output(
            model = model,
            img_path=image_path,
            prompt=prompt)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": output_text,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    

def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4-0314',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def eval_gpt4(args):
    print("evaluating with GPT4 now...")
    openai.api_key = args.openai_key
    os.makedirs(f"../LLaVA_benchmark/{args.model_name}", exist_ok=True)
    f_q = open(os.path.expanduser(args.question_file))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]
        cap_str = '\n'.join(inst['caption'])
        #box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

        category = json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()


def show_score(args):

    jsonl_file_path = args.output

    with open(jsonl_file_path, "r") as jsonl_file:
        lines = jsonl_file.readlines()

    categories = {
        'conv': {'num': 0, 'total_model': 0, 'total_gpt4': 0},
        'detail': {'num': 0, 'total_model': 0, 'total_gpt4': 0},
        'complex': {'num': 0, 'total_model': 0, 'total_gpt4': 0}
        
    }
    total_overall_model = 0
    total_overall_gpt4 = 0
    overall_num = 0

    for line in lines:
        json_data = json.loads(line)
        category = json_data["category"]
        r_model = json_data["tuple"][1]   # model
        r_gpt4 = json_data["tuple"][0]   # gpt4
        overall_num += 1
        total_overall_gpt4 += r_gpt4 
        total_overall_model += r_model
        
        categories[category]['num'] += 1
        categories[category]['total_model'] += r_model
        categories[category]['total_gpt4'] += r_gpt4

    print(f"evaluating file path:{args.model_name}")

    for cat, cat_data in categories.items():
        avg_model = cat_data['total_model'] / cat_data['num']
        avg_gpt4 = cat_data['total_gpt4'] / cat_data['num']
        print(f"###{cat}:###")
        print(f"{cat}_num:", cat_data['num'])
        print(f"avg_{cat}_model:", avg_model)
        print(f"avg_{cat}_gpt4:", avg_gpt4)
        print("final result:", avg_model / avg_gpt4 * 100)

    avg_overall_model = total_overall_model / overall_num
    avg_overall_gpt4 = total_overall_gpt4 / overall_num
    print("###overall:###")
    print("overall_num:", overall_num)
    print("avg_overall_model:", avg_overall_model)
    print("avg_overall_gpt4:", avg_overall_gpt4)
    print("final result:", avg_overall_model / avg_overall_gpt4 * 100)



args = get_args_parser().parse_args()  

'''
    inference: Get model answers.
    eval: Use GPT4 to score the mod's answers against the GPT4 answers.
    show: Output of the scored results
    all: Inferring, scoring, and outputting results for models.
'''
if args.mode == "inference":
    model = load(args)
    eval_llava_benchmark(model, args)
elif args.mode == "eval": 
    eval_gpt4(args)
    show_score(args)
elif args.mode == "show":
    show_score(args)
elif args.mode == "all":
    model = load(args)
    eval_llava_benchmark(model, args)
    eval_gpt4(args)
    show_score(args)


