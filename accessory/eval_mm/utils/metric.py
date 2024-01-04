"""
A Metric observes output of certain model, for example, in form of logits or
scores, and accumulates a particular metric with reference to some provided
targets. In context of VisDial, we use Recall (@ 1, 5, 10), Mean Rank, Mean
Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

Each ``Metric`` must atleast implement three methods:
    - ``observe``, update accumulated metric with currently observed outputs
      and targets.
    - ``retrieve`` to return the accumulated metric., an optionally reset
      internally accumulated metric (this is commonly done between two epochs
      after validation).
    - ``reset`` to explicitly reset the internally accumulated metric.

Caveat, if you wish to implement your own class of Metric, make sure you call
``detach`` on output tensors (like logits), else it will cause memory leaks.

Credit:
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/metrics.py
"""
import torch
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Tuple, List, Optional

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks


class SparseGTMetrics(object):
    """
    A class to accumulate all metrics with sparse ground truth annotations.
    These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
    """

    def __init__(self):
        self._rank_list = []

    def observe(
            self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
    ):
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()

        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]
        self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))

    def retrieve(self, reset: bool = True, key=""):
        num_examples = len(self._rank_list)
        if num_examples > 0:
            # convert to numpy array for easy calculation.
            __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                key + "r@1": torch.mean((__rank_list <= 1).float()).item(),
                key + "r@5": torch.mean((__rank_list <= 5).float()).item(),
                key + "r@10": torch.mean((__rank_list <= 10).float()).item(),
                key + "mean": torch.mean(__rank_list).item(),
                key + "mrr": torch.mean(__rank_list.reciprocal()).item(),
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._rank_list = []


class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
            self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze(1)
        batch_size, num_options = predicted_ranks.size()

        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True, key=""):
        if self._ndcg_denominator > 0:
            metrics = {
                key + "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0



# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(
            prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], list):
            score = max([relaxed_correctness(elem['answer'].strip(), ann) for ann in elem['annotation']])
        else:
            score = relaxed_correctness(elem['answer'].strip(), elem['annotation'])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        pred = elem['answer'].replace('.', '')
        gt = elem['annotation'].replace('.', '')
        if isinstance(elem['annotation'], list):
            score = max(
                [(1.0 if (pred.strip().lower() == ann.strip().lower()) else 0.0) for ann in
                 gt])
        else:
            score = pred.strip().lower() == gt.strip().lower()
        scores.append(score)
    return sum(scores) / len(scores)


def parse_pred_ans(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"

    return pred_label


def compute_mme_metric(gts, preds):
    assert len(gts) == len(preds)

    label_map = {
        "yes": 1,
        "no": 0,
        "other": -1,
    }

    gts = [label_map[x] for x in gts]
    preds = [label_map[x] for x in preds]

    acc = accuracy_score(gts, preds)

    clean_gts = []
    clean_preds = []
    other_num = 0
    for gt, pred in zip(gts, preds):
        if pred == -1:
            other_num += 1
            continue
        clean_gts.append(gt)
        clean_preds.append(pred)

    conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
    precision = precision_score(clean_gts, clean_preds, average='binary')
    recall = recall_score(clean_gts, clean_preds, average='binary')
    tp, fn = conf_mat[0]
    fp, tn = conf_mat[1]

    metric_dict = dict()
    metric_dict = {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "precision": precision,
        "recall": recall,
        "other_num": other_num,
        "acc": acc,
    }

    return metric_dict