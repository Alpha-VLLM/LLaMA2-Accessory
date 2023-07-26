import torch
import torch.nn as nn
import json

from fairscale.nn.model_parallel import initialize as fs_init

from .tokenizer import Tokenizer
from . import LLM
from util import misc


class MetaModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
        self, llama_type: str, llama_config: str, tokenizer_path: str,
        with_visual: bool = False, max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        ModelArgs = LLM.__dict__[llama_type].ModelArgs
        Transformer = LLM.__dict__[llama_type].Transformer

        with open(llama_config, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=32, **params
        )
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        model = Transformer(model_args, with_visual=with_visual)
        self.llma = model

        misc.mark_mp_params(self)

        param_count_local, param_count_all = 0, 0
        for name, param in self.named_parameters():
            is_model_parallel = getattr(param, "is_model_parallel", False)
            if param.requires_grad:
                print(f"Trainable param: {name}, local_size: {param.shape}, model_parallel: {is_model_parallel}, dtype: {param.dtype}")
                if is_model_parallel:
                    param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    param_count_all += param.numel()
                param_count_local += param.numel()
        print(f"Parameter count : {param_count_local} (local rank), {param_count_all} (all).")


    def forward(self, examples, labels, images=None):
        output = self.llma(examples, images)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
           c_loss = output.mean() * 0
        else:
           c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())
        return c_loss


    def generate(
        self,
        prompts: list[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(tokens[:, :cur_pos], 0, images)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


    def get_image_words(self):
        return self.llma.image_words
