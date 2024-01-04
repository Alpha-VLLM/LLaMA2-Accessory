import random
from typing import List
from tqdm import tqdm
import sys
import os
import multiprocessing as mp

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])
from evaluate import Evaluator
from sphinx import SPHINXModel

from data.conversation.lib import conv_templates, SeparatorStyle
import argparse
import torch
import torch.distributed as dist
from PIL import Image
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize
from util.misc import setup_for_distributed
import json
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    image_path = [_['image_path'] for _ in batches]
    raw_question = [_['question_raw'] for _ in batches]

    input_image = torch.cat([_['image'] for _ in batches])

    return input_image, question_ids, questions, annotations, image_path, raw_question


class PadToSquare:
    def __init__(self, background_color):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x * 255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result


def T_padded_resize(size=224):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, anno_path, prompt, img_root, img_size=224):
        with open(anno_path, 'r') as f:
            self.annotation = json.loads(f.read())
        # split data for different rank
        self.prompt = prompt
        self.img_root = img_root
        self.transform_val = T_padded_resize(img_size)
        self.multiple_choices = ['A', 'B', 'C', 'D', 'E']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        data = self.annotation[idx]
        image, question, question_id, annotation, hint, choices = data['image_path'], data[
            'question'], data.get('question_id', -1), data.get('gt_answers', ''), data.get('hint', 'N/A'), data.get(
            'choices', None),

        image = Image.open(image).convert('RGB')
        image = self.transform_val(image).unsqueeze(0)

        conv = conv_templates["v1"].copy()
        input_prompt = self.prompt['input']
        output_prompt = self.prompt['output']
        if input_prompt.count('{}') > 1:
            # for multiple choice VQA
            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append('{}. {}'.format(self.multiple_choices[i], c))
            choice_txt = '\n'.join(choice_list)
            input_text = input_prompt.format(hint, question, choice_txt)
        else:
            # for traditional VQA
            input_text = input_prompt.format(question)

        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        question = conv.get_prompt()
        question = question + output_prompt

        return {
            'question': question,
            'question_id': question_id,
            'annotation': annotation,
            'image': image,
            'image_path': data['image_path'],
            'question_raw': data['question']
        }


def get_local_indices(rank: int, world_size: int, dataset_len: int) -> List[int]:
    indices = list(range(dataset_len))
    # there exist duplication in data
    while len(indices) % world_size != 0:
        indices.extend(indices[: world_size - len(indices) % world_size])
    indices = indices[rank::world_size]
    return indices


def main(args, rank: int, world_size: int, master_port: int, master_addr: str,
         model_parallel_size: int):
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("nccl")
    setup_for_distributed(dist.get_rank() == 0)
    fs_init.initialize_model_parallel(model_parallel_size)
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # define the model
    model = SPHINXModel.from_pretrained(
        pretrined_path=args.pretrained_path, with_visual=True,
        mp_group=fs_init.get_model_parallel_group()
    )

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

    with open('./annotations/annotation_config.json', 'r') as f:
        ds_collections = json.loads(f.read())

    if args.dataset[0] == 'all':
        dataset_names = ds_collections.keys()
    else:
        dataset_names = args.dataset

    for ds in dataset_names:
        base_result_dir = f'results/{args.pretrained_path[0].split("ckpts")[-1].replace("/", "_")}'
        log_dir = f'{base_result_dir}/results.txt'
        if os.path.exists(log_dir):
            with open(log_dir, 'r') as f:
                pre_log = f.read()
            if ds in pre_log:
                print(f'Dataset: {ds} is tested, skip here.')
                continue

        if 'prompt' in ds_collections[ds]:
            prompt = ds_collections[ds]['prompt']
        else:
            prompt = {
                'input': '{}',
                'output': ''
            }
        random.seed(args.seed)
        dataset = VQADataset(
            anno_path=ds_collections[ds]['test'],
            img_root=args.img_root,
            prompt=prompt,
            img_size=getattr(model.llma, 'image_size', 224)
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=get_local_indices(dist.get_rank() // model_parallel_size, fs_init.get_data_parallel_world_size(),
                                      len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        conv = conv_templates["v1"].copy()
        conv_sep = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
        outputs = []
        max_gen_len = ds_collections[ds]['max_new_tokens']
        gen_t = args.temperature
        top_p = args.top_p

        global_config = {
            'gen_t': gen_t,
            'top_p': top_p,
            'max_gen_len': max_gen_len
        }
        with torch.no_grad():
            for image, question_ids, _prompt, annotations, image_path, raw_question in tqdm(dataloader,
                                                                                            desc=f'{ds}: {len(dataset)} samples'):
                if dist.get_rank() % model_parallel_size == 0:
                    dist.barrier()
                    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p],
                                               src=fs_init.get_model_parallel_src_rank(),
                                               group=fs_init.get_model_parallel_group(),
                                               )

                    image = image.cuda(non_blocking=True)
                    print(
                        f'\nrank: {rank} mp rank: {fs_init.get_model_parallel_rank()} img_path: {image_path[0]} input: {_prompt[0]}\n')

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        results = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t,
                                                 top_p=top_p, additional_stop_symbols=['###'])

                    for question_id, answer, annotation, img_path, r_ques in zip(question_ids, results,
                                                                                 annotations, image_path, raw_question):

                        end_pos = answer.find(conv_sep)
                        if end_pos != -1:
                            answer = answer[:end_pos].rstrip()

                        outputs.append(
                            {'image_path': img_path, 'question': r_ques, 'answer': answer, 'gt_answers': annotation,
                             'question_id': question_id}
                        )
                    print(
                        f'\nrank: {rank} mp rank: {fs_init.get_model_parallel_rank()}  gt: {annotation} pred: {answer}\n')
                else:
                    dist.barrier()

                    input_data = [None for _ in range(5)]
                    dist.broadcast_object_list(input_data,
                                               src=fs_init.get_model_parallel_src_rank(),
                                               group=fs_init.get_model_parallel_group(),
                                               )
                    _prompt, image, max_gen_len, gen_t, top_p = input_data
                    image = image.cuda(non_blocking=True)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                           additional_stop_symbols=['###'])

        torch.distributed.barrier()
        if fs_init.get_data_parallel_world_size() > 1:
            outputs_allgather = [None for _ in range(fs_init.get_data_parallel_world_size())]
            dist.all_gather_object(outputs_allgather, outputs, fs_init.get_data_parallel_group())
            outputs = list(sum(zip(*outputs_allgather), ()))


        if torch.distributed.get_rank() == 0:
            # remove duplication from padded data sampler
            cleaned_output = {}
            for p_i in outputs:
                key = f'{p_i["image_path"]}_{p_i["question"]}_{p_i["question_id"]}'
                if key not in cleaned_output:
                    cleaned_output[key] = p_i

            outputs = list(cleaned_output.values())

            # evaluate result and save logs
            evaluator = Evaluator(ds_collections, global_config, prompt)
            evaluator.evaluate(outputs, ds, args)

        torch.distributed.barrier()


if __name__ == '__main__':

    def get_args_parser():
        parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
        # Model parameters
        parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                            help='Path to llama model config')
        parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                            help='path to tokenizer.model')
        parser.add_argument('--img_root', type=str, default="./data/nocaps/images",
                            help='path to tokenizer.model')
        parser.add_argument('--annotation_path', type=str, default="./data/nocaps/nocap_val.json",
                            help='path to tokenizer.model')

        parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')

        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)

        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument('--quant', action="store_true", default=False,
                            help="enable quantization")
        parser.add_argument('--dataset', default='vqav2_val', type=str, nargs="+")
        parser.add_argument("--max_seq_length", type=int, default=2048)
        parser.add_argument("--master_port", type=int, default=None)
        parser.add_argument("--master_addr", type=str, default="127.0.0.1")
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--top_p", type=float, default=0.75)

        return parser


    args = get_args_parser().parse_args()

    torch.multiprocessing.set_start_method('spawn')

    if args.master_port is None:
        args.master_port = random.randint(20000, 30000)

    gpu_procs = []
    print(f"Using {torch.cuda.device_count()} gpu(s).")
    for i in range(torch.cuda.device_count()):
        try:
            p = mp.Process(
                target=main,
                args=(args, i, torch.cuda.device_count(), args.master_port, args.master_addr,
                      args.model_parallel_size),
            )
            p.start()
            gpu_procs.append(p)
        except Exception as e:
            print(e)

    for p in gpu_procs:
        p.join()
