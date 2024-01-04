import itertools
import random
import time
from functools import partial
from typing import Optional, List, Tuple
from tqdm import tqdm
from utils.vqa import VQA
from utils.vqa_eval import VQAEval
from utils.utils import save_result
from utils.metric import relaxed_correctness, evaluate_relaxed_accuracy, evaluate_exact_match_accuracy, \
    compute_mme_metric, parse_pred_ans
import sys
import os
import re
from utils.math_utils import compare_both_string_and_number_format
import multiprocessing as mp

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
from util.tensor_type import default_tensor_type
from model.meta import MetaModel
from data.conversation.lib import conv_templates, SeparatorStyle
import argparse
import torch
import torch.distributed as dist
from PIL import Image
import PIL.ImageFile as ImageFile

from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize
from util.misc import setup_for_distributed
import json
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_area

# Increase the limit for decompression
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb limit

DUE_BENCHMARK_Datasets = ["DeepForm", "InfographicsVQA", "KleisterCharity", "TabFact", "WikiTableQuestions", "ChartQA", "VisualMRC"]


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


class Evaluator:
    def __init__(self, config, global_config, prompt):
        self.config = config
        self.global_config = global_config
        self.prompt = prompt
    
    def evaluate(self, outputs, ds, args):
        base_result_dir = f'results/{args.pretrained_path[0].split("ckpts")[-1].replace("/", "_")}'
        os.makedirs(base_result_dir, exist_ok=True)
        os.makedirs('vqa_logs', exist_ok=True)
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{base_result_dir}/{ds}_{time_prefix}_{args.seed}.json'

        if self.config[ds]['metric'] == 'vqa_score':
            vqa = VQA(self.config[ds]['annotation'],
                      self.config[ds]['question'])

            json.dump(outputs, open(results_file, 'w'),
                      ensure_ascii=False)
            results = vqa.loadRes(
                resFile=results_file,
                quesFile=self.config[ds]['question'])
            vqa_scorer = VQAEval(vqa, results, n=2)
            vqa_scorer.evaluate()

            print(vqa_scorer.accuracy)
            save_result(args, vqa_scorer.accuracy, self.prompt, self.global_config, self.config, results_file, ds)
        elif self.config[ds]['metric'] == 'mme_score':
            base_mme_dir = f'{base_result_dir}/MME_results'
            os.makedirs(base_mme_dir, exist_ok=True)
            # MME evaluation
            eval_type_dict = {
                "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene",
                               "landmark", "artwork", "OCR"],
                "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation",
                              "code_reasoning"]
            }
            pred_by_category = {}
            for pred in outputs:
                cate = None
                for c in eval_type_dict['Perception'] + eval_type_dict['Cognition']:
                    if c in pred['image_path']:
                        cate = c
                if cate is None:
                    raise ValueError

                if cate not in pred_by_category:
                    pred_by_category[cate] = [pred]
                else:
                    pred_by_category[cate].append(pred)

            for k, v in pred_by_category.items():
                v.sort(key=lambda x: x['question_id'])

                out_datas = [
                    f"{data['image_path']}\t{data['question']}\t{data['gt_answers']}\t{data['answer']}"
                    for data in v
                ]
                with open(f'{base_mme_dir}/{k}.txt', 'w') as f:
                    f.write('\n'.join(out_datas))

            result = {}
            for dataset_name, dataset_pred in pred_by_category.items():
                gts = []
                preds = []
                img_correct_num = 0
                task_other_ans_num = 0
                acc_plus_correct_num = 0
                last_correct = False
                for i in range(len(dataset_pred)):
                    gt_answers = dataset_pred[i]['gt_answers'].lower()
                    answer = dataset_pred[i]['answer'].lower()
                    assert gt_answers in ['yes', 'no']
                    answer = parse_pred_ans(answer)
                    gts.append(gt_answers)
                    preds.append(answer)

                    if gt_answers == answer:
                        img_correct_num += 1
                        if (i + 1) % 2 == 0 and last_correct:
                            acc_plus_correct_num += 1
                        last_correct = True
                    else:
                        last_correct = False

                    if answer == 'other':
                        task_other_ans_num += 1

                metric_dict = compute_mme_metric(gts, preds)
                metric_dict['acc_plus'] = (acc_plus_correct_num) / len(dataset_pred) * 2
                task_score = 0
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v * 100

                result[dataset_name] = task_score

            with open(f'{base_result_dir}/results.txt', 'a') as f:
                f.write('-' * 10 + '\n')
                f.write(f'mme result\n')
                f.write(json.dumps(result, indent=4))
                Perception_score = sum([result[i] for i in eval_type_dict['Perception']])
                Cognition_score = sum([result[i] for i in eval_type_dict['Cognition']])
                f.write(f'Perception_score: {Perception_score}\n')
                f.write(f'Cognition_score: {Cognition_score}\n')
                print(f'Perception_score: {Perception_score}\n')
                print(f'Cognition_score: {Cognition_score}\n')
                f.write(f'Sum_score: {Perception_score + Cognition_score}\n')
                f.write('-' * 10 + '\n')
        elif self.config[ds]['metric'] == 'anls':
            format_result = []
            for pred in outputs:
                format_result.append({
                    'questionId': pred['question_id'],
                    'answer': pred['answer'],
                    'annotation': pred['gt_answers'],
                })
            json.dump(format_result, open(results_file, 'w'),
                      ensure_ascii=False)
            # results_file = f'vqa_log/{ds}_{time_prefix}.json'
            # json.dump(outputs, open(results_file, 'w'), ensure_ascii=False)
            print('python infographicsvqa_eval.py -g ' + self.config[ds][
                'annotation'] + ' -s ' + results_file)
            os.system('python infographicsvqa_eval.py -g ' + self.config[ds][
                'annotation'] + ' -s ' + results_file)

            save_result(args, 'python infographicsvqa_eval.py -g ' + self.config[ds][
                'annotation'] + ' -s ' + results_file, self.prompt, self.global_config, self.config, results_file, ds)
        elif self.config[ds]['metric'] == 'relaxed_accuracy':
            format_result = []
            for pred in outputs:
                format_result.append({
                    'annotation': pred['gt_answers'],
                    'answer': pred['answer']
                })
            json.dump(format_result, open(results_file, 'w'),
                      ensure_ascii=False)

            print({'relaxed_accuracy': evaluate_relaxed_accuracy(format_result)})
            save_result(args, {'relaxed_accuracy': evaluate_relaxed_accuracy(format_result)}, self.prompt, self.global_config,
                        self.config, results_file, ds)
        elif self.config[ds]['metric'] == 'accuracy':
            format_result = []
            multiple_choices = ['A', 'B', 'C', 'D', 'E']
            for pred in outputs:
                answer = pred['answer']
                if pred['gt_answers'] in multiple_choices:
                    answer = answer.split('.')[0]
                    if answer not in multiple_choices:
                        print(f'not matched answer: {answer}')
                        answer = random.choice(multiple_choices)
                format_result.append({
                    'questionId': pred['question_id'],
                    'answer': answer,
                    'annotation': pred['gt_answers'],
                })
            json.dump(format_result, open(results_file, 'w'),
                      ensure_ascii=False)

            # if 'seedbenchv2' in ds:
            #     l1_format_result = [i for i in format_result if i['questionId'].split('||')[1] == 'L1']
            #     l1_format_result = [i for i in format_result if i['questionId'].split('||')[1] == 'L1']

            if 'gqa' in ds:
                for entry in outputs:
                    response = entry['answer']
                    response = response.strip().split('.')[0].split(',')[0].split('!')[0].lower()
                    if 'is ' in response:
                        response = response.split('is ')[1]
                    if 'are ' in response:
                        response = response.split('are ')[1]
                    if 'a ' in response:
                        response = response.split('a ')[1]
                    if 'an ' in response:
                        response = response.split('an ')[1]
                    if 'the ' in response:
                        response = response.split('the ')[1]
                    if ' of' in response:
                        response = response.split(' of')[0]
                    response = response.strip()
                    entry['answer'] = response

            print({'accuracy': evaluate_exact_match_accuracy(format_result)})
            save_result(args, {'accuracy': evaluate_exact_match_accuracy(format_result)}, self.prompt, self.global_config,
                        self.config, results_file, ds)
        elif self.config[ds]['metric'] == 'wenqi':
            format_result = []
            for pred in outputs:
                format_result.append({
                    'task_name': pred['question_id'],
                    'answer': pred['answer'],
                })
            json.dump(format_result, open(results_file, 'w'),
                      ensure_ascii=False)

        elif self.config[ds]['metric'] == 'mmmu_save':
            base_mmmu_dir = f'{base_result_dir}/MMMU_results'
            os.makedirs(base_mmmu_dir, exist_ok=True)
            format_result = {}
            for pred in outputs:
                ans = pred['answer']
                if ans.endswith('.'):
                    ans = ans[:-1]
                format_result[pred['question_id'].split('||')[0]] = ans

            json.dump(format_result, open(os.path.join(base_mmmu_dir, f'{ds}.json'), 'w'),
                      ensure_ascii=False)
            
        elif self.config[ds]['metric'] == 'ureader':
            base_ureader_dir = f'{base_result_dir}/Ureader_results'
            os.makedirs(base_ureader_dir, exist_ok=True)
            format_result = []
            for pred in outputs:
                format_result.append({
                    'question_id': pred['question_id'],
                    'answer': pred['answer'],
                    "annotation": pred['gt_answers'],
                    "question": pred.get('question', None),
                    "name": pred.get('image_path', None),
                })
            print(f'{ds} results saved to {results_file}')
            save_path = f'{base_ureader_dir}/{ds}.json'
            json.dump(format_result, open(save_path, 'w'),
                      ensure_ascii=False)
            save_result(args, f'python ureader_eval/tools.py --eval_file_folder {base_ureader_dir}', self.prompt, self.global_config, self.config, results_file, ds)

        elif self.config[ds]['metric'] == 'miou0.5':
            json.dump(outputs, open(results_file, 'w'),
                      ensure_ascii=False)

            correct = total_cnt = 0
            PATTERN = re.compile(r'\[(.*?)\]')
            for pred in outputs:
                predict_bbox = re.findall(PATTERN, pred['answer'])

                if predict_bbox == []:
                    predict_bbox = [pred['answer'].split(']')[0]]
                    # print(predict_bbox)

                try:
                    tmp_ans = predict_bbox[0].split(',')
                    if len(tmp_ans) != 4:
                        predict_bbox = (0., 0., 0., 0.)
                    else:
                        predict_bbox = [float(tmp) for tmp in tmp_ans]
                except:
                    predict_bbox = (0., 0., 0., 0.)

                height = int(re.search(r'height(\d+)', pred["question_id"]).group(1))
                width = int(re.search(r'width(\d+)', pred["question_id"]).group(1))
                max_edge = max(height, width)

                gt_box = pred["gt_answers"]
                gt_box_padded = box_xyxy_expand2square(gt_box, h=height, w=width)
                target_bbox = torch.tensor(gt_box_padded,
                                           dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(predict_bbox,
                                            dtype=torch.float32).view(-1, 4)
                predict_bbox[:, 0::2] *= max_edge
                predict_bbox[:, 1::2] *= max_edge

                iou, _ = box_iou(predict_bbox, target_bbox)
                iou = iou.item()
                total_cnt += 1
                if iou >= 0.5:
                    correct += 1

            print(f'Precision @ 1: {correct / total_cnt} \n')
            save_result(args, {'miou0.5 acc:': correct / total_cnt}, self.prompt,
                        self.global_config,
                        self.config, results_file, ds)
        elif self.config[ds]['metric'] == 'MATH':
            correct, wrong = 0, 0
            for pred in outputs:
                answer = pred['answer']
                groundtruth = json.loads(pred['gt_answers'])

                groundtruth_str, groundtruth_num = groundtruth
                if compare_both_string_and_number_format(answer, groundtruth_str, groundtruth_num):
                    correct += 1
                else:
                    wrong += 1

            json.dump(outputs, open(results_file, 'w'),
                      ensure_ascii=False)

            print({'accuracy': correct / (correct + wrong)})
            save_result(args, {'accuracy': correct / (correct + wrong)}, self.prompt,
                        self.global_config,
                        self.config, results_file, ds)
            
            

