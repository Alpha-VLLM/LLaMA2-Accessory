import warnings
import os
import torch
import torch.nn as nn
import json
from typing import List, Dict, Optional, Iterable
from pathlib import Path
import inspect
import importlib

from fairscale.nn.model_parallel import initialize as fs_init

from .tokenizer import Tokenizer, probe_tokenizer_path_from_pretrained
from accessory.util import misc, tensor_parallel
from accessory.util.tensor_type import default_tensor_type
import torch.distributed as dist


class MetaModel(nn.Module):
    def __init__(
        self, llama_type: str, llama_config: str|List[str], tokenizer_path: str,
        with_visual: bool = False, max_seq_len: int = 4096
    ) -> None:
        super().__init__()

        self.llama_type = llama_type
        self.with_visual = with_visual

        model_module = importlib.import_module(f"accessory.model.LLM.{llama_type}")
        ModelArgs = model_module.ModelArgs
        Transformer = model_module.Transformer

        llama_args = {}
        if isinstance(llama_config, str):
            llama_config = [llama_config]
        for _ in llama_config:
            with open(_, "r") as f:
                llama_args.update(json.loads(f.read()))
        llama_args['max_seq_len'] = max_seq_len
        llama_args['max_batch_size'] = 32

        tokenizer = Tokenizer(model_path=tokenizer_path)
        llama_args['vocab_size'] = tokenizer.n_words

        llama_args: ModelArgs = ModelArgs(**llama_args)

        if "tokenizer" in inspect.signature(Transformer.__init__).parameters:
            # generally it means the inner llm modify change the tokenizer
            model = Transformer(llama_args, tokenizer, with_visual=with_visual)
            assert hasattr(model, "tokenizer")
            self.tokenizer = model.tokenizer
        else:
            model = Transformer(llama_args, with_visual=with_visual)
            self.tokenizer = tokenizer

        print("Model Args:\n", model.args)

        self.llma = model

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._set_default_trainability()

        self.is_peft = getattr(model, "is_peft", False)
        print(f"Model is Peft: {self.is_peft}")

        misc.mark_mp_params(self)

        param_count_local, param_count_all = 0, 0
        for name, param in self.named_parameters():
            is_model_parallel = getattr(param, "is_model_parallel", False)
            if param.requires_grad:
                if is_model_parallel:
                    param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    param_count_all += param.numel()
                param_count_local += param.numel()
        print(f"Trainable parameter count : {param_count_local} (local rank), {param_count_all} (all).")

    @classmethod
    def from_pretrained(cls, pretrained_path: str|List[str],
                        llama_type: Optional[str] = None,
                        llama_config: Optional[str|List[str]] = None,
                        tokenizer_path: Optional[str] = None,
                        with_visual: bool = False, max_seq_len: int = 4096,
                        mp_group: Optional[dist.ProcessGroup] = None,
                        dtype=torch.bfloat16, device="cuda", quant=False):
        """
        Besides loading the `consolidated.*.pth` model weights, this function also tries to find tokenizer,
        'meta.json', and 'config.json' under `pretrained_path` to configure the `tokenizer_path`,
        `llama_type`, and `llama_config` of the model. The automatically determined values will be
        overridden by user's exploit specification of the arguments.
        :param pretrained_path: Paths to directories containing `consolidated.*.pth` weight files. If multiple paths
                are given, weights will be loaded sequentially. Now repo_id also can be specified as a path.
        :param llama_type: Type of the inner LLM. The corresponding model class definition should be found in
                accessory/model/LLM/llama_type.py. If not specified, this function will probe the `meta.json`
                file under `pretrained_path` to try to determine the value.
        :param llama_config: Inner LLM configurations. Can be one or a list of strings, each of which is the path
                to a `*.json` configuration file. If not specified, this function will probe the `config.json`
                file under `pretrained_path` to try to determine the value.
        :param tokenizer_path: LLaMA2-Accessory supports both spm tokenizers (provided by Meta, generally named
                `tokenizer.model`) and huggingface tokenizers (composed of tokenizer.json and tokenizer_config.json).
                When using spm tokenizers, tokenizer_path should point to the `tokenizer.model` file;
                when using huggingface tokenizers, tokenizer_path should point to the directory containing
                tokenizer.json and tokenizer_config.json. If not specified, this function will probe the
                `pretrained_path` directory for tokenizer in either format.
        :param with_visual: Set it to True if the model is expected to receive image input. Inner LLM models
                rely on this argument to decide whether to instantiate the visual encoder.
        :param max_seq_len: max context window size of the model
        :param mp_group:  If the parameters of the model are *not* split on multiple GPUs with model parallel,
                namely model parallel size == 1, then `mp_group` can be left to `None`. However, if model
                parallel is needed, `mp_group` should be an already initialized torch process group, ranks
                within which compose a logically complete model.
        :param dtype: parameter data type
        :param device: parameter device
        :param quant: whether to quantize the model to 4bit

        :return: An Accessory.model.MetaModel object with pretrained checkpoints loaded.
        """
        if isinstance(pretrained_path, str):
            pretrained_path = [pretrained_path]
        if pretrained_path is None or len(pretrained_path) == 0:
            raise ValueError("pretrained_path should be specified")

        for i, path in enumerate(pretrained_path):
            if path.startswith("hf://"):
                print(f"load {path} from huggingface...")
                cached_path = misc.cached_file_from_hf(path)
                pretrained_path[i] = cached_path
                print(f"{path} cached to {cached_path}")

        if mp_group is None:
            print(f"mp_group not provided. Load model with model parallel size == 1")
            if dist.is_initialized():
                mp_group = dist.new_group(ranks=[dist.get_rank()])
            else:
                warnings.warn(
                    "\n\n********************************\n"
                    "Warning: Torch distributed not initialized when invoking `MetaModel.from_pretrained`.\n"
                    "trying to init distributed mode within `from_pretrained` with a world size of 1.\n"
                    "Note: Distributed functions like `get_world_size()` are used within Accessory's model implementations,\n"
                    "Therefore, distributed initialization is required even when using a single GPU.\n"
                    "This warning can be ignored if your program isn't designed for distributed computing.\n"
                    "However, if your program also relies on the functionalities from `torch.distributed`,\n"
                    "please initialize distributed mode before model creation\n"
                    "********************************\n")
                torch.distributed.init_process_group(
                    backend="nccl", rank=0, world_size=1,
                    init_method=f"tcp://127.0.0.1:{misc.find_free_port(9000, 10000)}")
                mp_group = dist.new_group(ranks=[dist.get_rank()])
        else:
            print(f"Load model with model parallel size == {dist.get_world_size(mp_group)}")

        fs_init._MODEL_PARALLEL_GROUP = mp_group

        # determine llama_type
        if llama_type is None:
            print(f"llama_type not specified, attempting to obtain from {Path(pretrained_path[-1])/'meta.json'}")
            if (Path(pretrained_path[-1])/'meta.json').exists():
                with open(Path(pretrained_path[-1])/'meta.json', 'r') as f:
                    llama_type = json.load(f)["llama_type"]
                    print(f"Obtained llama_type: {llama_type}")
            else:
                print(f"{Path(pretrained_path[-1])/'meta.json'} does not exist")
                raise ValueError("Cannot determine llama_type")


        # determine llama_config
        if llama_config is None:
            print(f"llama_config not specified, attempting to find {Path(pretrained_path[-1]) / 'config.json'}")
            if (Path(pretrained_path[-1])/'config.json').exists():
                llama_config = [str(Path(pretrained_path[-1])/'config.json')]
                print(f"Found llama_config: {str(Path(pretrained_path[-1])/'config.json')}")
            else:
                print(f"{str(Path(pretrained_path[-1]) / 'config.json')} does not exist\n"
                      f"will use the default config values (specified in the definition of ModelArgs in {llama_type}.py)")
                llama_config = []


        # determine tokenizer_path
        if tokenizer_path is None:
            print(f"tokenizer_path not specified, probe from pretrained path {pretrained_path[-1]}")
            tokenizer_path = probe_tokenizer_path_from_pretrained(pretrained_path[-1])
            if tokenizer_path is None:
                raise FileNotFoundError("No tokenizer available")
            print(f"Use tokenizer_path: {tokenizer_path}")


        with default_tensor_type(dtype=dtype, device="cpu" if quant else device):
            model = cls(llama_type, llama_config, tokenizer_path, with_visual, max_seq_len)
        print(f"Loading pretrained weights from {pretrained_path} ...")
        load_result = tensor_parallel.load_tensor_parallel_model_list(model, pretrained_path)
        if load_result != {'missing_keys': [], 'unexpected_keys': []}:
            warnings.warn(f"checkpoint and model mismatch: \n{load_result}")
        else:
            print("all params match perfectly!")

        if quant:
            from accessory.util.quant import quantize
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
            model.to(device)

        model.eval()
        return model

    def get_trainable_params(self):
        llma_trainable = self.llma.get_trainable_params()
        return {"llma." + name: param for name, param in llma_trainable.items()}

    def _set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.requires_grad = True

    def forward(self, examples, labels, images=None):
        with torch.no_grad():
            non_zero_ = torch.count_nonzero(labels, dim=0)
            pos = non_zero_.shape[0] - 1
            while pos >= 0:
                if non_zero_[pos] == 0:
                    pos -= 1
                else:
                    break

            if pos == -1:  # nothing to predict in the whole batch
                print(f"[RANK {dist.get_rank()}] nothing to predict in the whole batch!", force=True)
                print(examples.cpu().tolist(), force=True)
                pos = 2
            examples = examples[:, :pos+1]
            labels = labels[:, :pos+1]

        output = self.llma(examples, images)
        if isinstance(output, tuple):
            output, additional_loss = output
        else:
            additional_loss = {}
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
           c_loss = output.mean() * 0
        else:
           c_loss = self.criterion(output.reshape(-1, self.tokenizer.n_words), labels.flatten())
        return c_loss, additional_loss

    @ torch.inference_mode()
    def compute_logits(self, examples: List[str|List[int]], images:Optional[torch.FloatTensor]=None,
                       bos=True, eos=False) -> List[torch.FloatTensor]:
        """
        Compute logits for a given list of text examples or token lists, optionally incorporating images.

        :param examples: A batched list of text examples or their encoded token lists to be processed.
        :param images: Optional; batched image tensor to be used in conjunction with the text examples.
         Shape: (bsz, channel, h, w).
        :param bos: Whether to include begin-of-sequence tokens for tokenization. Only effective when items
         in `examples` are strings. Default is True.
        :param eos: Whether to include end-of-sequence tokens for tokenization. Only effective when items
         in `examples` are strings. Default is False.
        :return: A list of `torch.FloatTensor` containing the computed logits for each example.
        """
        if isinstance(examples, str):
            raise ValueError(f"{self.__class__}.generate expects a batched LIST of prompts, but str is given")

        if isinstance(examples[0], str):
            examples = [self.tokenizer.encode(_, bos, eos) for _ in examples]

        if images is not None:
            images = images.to(list(self.parameters())[0].device)

        l_seq_len = [len(_) for _ in examples]
        bsz = len(examples)
        max_length = max(l_seq_len)

        token_tensor = torch.full((bsz, max_length), 0).cuda().long()
        for i, item_tokens in enumerate(examples):
            token_tensor[i, : len(item_tokens)] = torch.tensor(item_tokens).long()

        output = self.llma(token_tensor, images)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        logits = [_[:seq_len].float() for _, seq_len in zip(logits, l_seq_len)]
        return logits

    @ torch.inference_mode()
    def evaluate_examples(self, examples: List[str|List[int]], contexts: Optional[List[str|List[int]]] = None,
                          images:Optional[torch.FloatTensor]=None, bos=True, eos=False) -> Dict[str, List]:
        """
        Evaluate text examples with optional contexts and images, returning various evaluation metrics.


        :param examples: A batched list of text examples or their encoded token lists.
        :param contexts: Optional; a list of context strings or token lists. If not None, each item
         should be the preceding part of the corresponding example and is considered as context.
         The calculation of evaluation metrics will be conducted only on the remaining part
         of the examples.
        :param images: Optional; batched image tensor to be used in conjunction with the text examples.
         Shape: (bsz, channel, h, w).
        :param bos: Whether to include begin-of-sequence tokens for tokenization. Only effective when items
         in `examples` are strings. Default is True.
        :param eos: Whether to include end-of-sequence tokens for tokenization. Only effective when items
         in `examples` are strings. Default is False.
        :return: A dictionary containing evaluation metrics including log likelihood, perplexity, max_equal,
        and non_content_logits.

        :Examples:
        >>> model = MetaModel(...)
        >>> # evaluate on the entire examples
        >>> model.evaluate_examples(["The best programming language is C", "The best programming language is Python"])
        >>> # treat "The best programming language is" as context and only evaluate on " C" and " Python"
        >>> model.evaluate_examples(
        >>>     examples=["The best programming language is C", "The best programming language is Python"],
        >>>     contexts=["The best programming language is", "The best programming language is"]
        >>> )
        """
        if isinstance(examples, str):
            raise ValueError(f"{self.__class__}.generate expects a batched LIST of prompts, but str is given")

        if isinstance(examples[0], str):
            examples = [self.tokenizer.encode(_, bos, eos) for _ in examples]
            if contexts is not None:
                contexts = [self.tokenizer.encode(_, bos, False) for _ in contexts]
        if contexts is not None:
            # when context is not None, `example == context + output` should hold,
            # namely example should start with context
            assert all([e[:len(c)] == c for e, c in zip(examples, contexts)])

        if images is not None:
            images = images.to(list(self.parameters())[0].device)

        logits = self.compute_logits(examples, images)

        loss_func = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        result = {'log_likelihood': [], 'ppl': [], 'max_equal': [], 'non_context_logits': []}
        for item_idx, item_logits in enumerate(logits):
            if contexts is None:
                logits_start = 0
            else:
                logits_start = len(contexts[item_idx]) - 1
                assert logits_start >= 0

            item_logits = item_logits[logits_start:-1]
            item_labels = examples[item_idx][logits_start+1:]
            item_labels = torch.tensor(item_labels, dtype=torch.long, device=item_logits.device)
            loss = loss_func(item_logits, item_labels)

            log_likelihood = -loss.sum().item()
            ppl = loss.mean().item()
            max_equal = (item_logits.argmax(dim=-1) == item_labels).all().item()

            result['log_likelihood'].append(log_likelihood)
            result['ppl'].append(ppl)
            result['max_equal'].append(max_equal)
            result['non_context_logits'].append(item_logits)
        return result

    @ torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        images: Optional[torch.FloatTensor] = None,
        max_gen_len: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        additional_stop_symbols: Iterable[str] = ()
    ) -> List[str]:
        """
        Generate text responses based on input prompts, optionally using images and controlling generation parameters.

        :param prompts: A batched list of string prompts for text generation.
        :param images: Optional; batched image tensor to be used in conjunction with the text examples.
         Shape: (bsz, channel, h, w).
        :param max_gen_len: Maximum generation length for the responses. Default is 512.
        :param temperature: Controls randomness in generation. Higher values lead to more random outputs.
         Default is 0.0, namely deterministic generation.
        :param top_p: Top-p sampling probability for more diverse generation. Default is 0.95.
        :param additional_stop_symbols: Iterable of additional symbols to stop generation.
        :return: A list of generated text responses corresponding to each input prompt.
        """

        if isinstance(prompts, str):
            raise ValueError(f"{self.__class__}.generate expects a batched LIST of prompts, but str is given")

        if images is not None:
            images = images.to(list(self.parameters())[0].device)

        bsz = len(prompts)
        args = self.llma.args
        assert bsz <= args.max_batch_size, (bsz, args.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        max_seq_len = args.max_seq_len
        if images is not None:
            max_seq_len -= self.llma.image_words

        total_len = min(max_seq_len, max_gen_len + max_prompt_size)
        for k, t in enumerate(prompt_tokens):
            prompt_tokens[k] = t[-(max_seq_len-max_gen_len):]

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        input_text_mask = torch.full((bsz, total_len), False).cuda()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask[k, : len(t)] = True
        start_pos = min_prompt_size
        prev_pos = 0

        l_stop_tokens = [[self.tokenizer.eos_id]]
        l_stop_tokens += [self.tokenizer.encode_segment(_) for _ in additional_stop_symbols]
        l_stop_tokens += [self.tokenizer.encode_wo_prefix_space(_) for _ in additional_stop_symbols]
        l_stop_tokens = [torch.tensor(_, dtype=tokens.dtype, device=tokens.device) for _ in l_stop_tokens]
        stopped = torch.tensor([False for _ in range(bsz)], device=input_text_mask.device)
        stop_pos = torch.tensor([start_pos + 1 for _ in range(bsz)], device=input_text_mask.device)
    
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(
                tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None
            ).float()
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

            stop_pos = torch.where(stopped, stop_pos, cur_pos + 1)
            for stop_token in l_stop_tokens:
                if cur_pos + 1 - len(stop_token) >= 0:
                    cond1 = (tokens[:, cur_pos+1-len(stop_token):cur_pos+1] == stop_token.unsqueeze(0)).all(dim=-1)
                    cond2 = ~input_text_mask[:, cur_pos]
                    new_stop_cond = cond1 * cond2 * (~stopped)
                    stop_pos = torch.where(new_stop_cond, cur_pos+1-len(stop_token), stop_pos)
                    stopped = torch.logical_or(new_stop_cond, stopped)
            if stopped.all():
                break

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = t[len(prompt_tokens[i]):stop_pos[i].item()]
            decoded.append(self.tokenizer.decode(t))
        return decoded

    @ torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        image: Optional[torch.FloatTensor] = None,
        max_gen_len: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        additional_stop_symbols: Iterable[str] = ()
    ):
        """
        Generate text in a streaming manner for a single prompt, optionally using an image.

        :param prompt: The input text prompt for generation.
        :param image: Optional; an image tensor to be used in conjunction with the text prompt.
         Shape: (channel, h, w) or (1, channel, h, w).
        :param max_gen_len: Maximum length for the generated text. Default is 512.
        :param temperature: Temperature for generation randomness. Default is 0.0,
         namely deterministic generation.
        :param top_p: Top-p sampling probability for diverse generation. Default is 0.95.
        :param additional_stop_symbols: Iterable of additional symbols to stop generation.
        :return: A generator yielding dictionaries with generated text and an end-of-content flag.
        """
        args = self.llma.args

        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        # truncate from the left. leave some space for generation.
        max_seq_len = args.max_seq_len
        if image is not None:
            max_seq_len -= self.llma.image_words
            if len(image.shape) == 4:
                assert image.shape[0] == 1
            else:
                assert len(image.shape) == 3
                image = image.unsqueeze(0)
            image = image.to(list(self.parameters())[0].device)

        max_prompt_size = max_seq_len - max_gen_len
        prompt_tokens = prompt_tokens[-max_prompt_size:]

        prompt_size = len(prompt_tokens)

        total_len = min(max_seq_len, max_gen_len + prompt_size)

        tokens = torch.full([total_len], 0).cuda().long()

        tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        start_pos = prompt_size
        prev_pos = 0
        generate_until = start_pos
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(
                tokens[None, prev_pos:cur_pos], prev_pos, image if prev_pos == 0 else None
            ).float()
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.item()

            if next_token == self.tokenizer.eos_id:
                break

            tokens[cur_pos] = next_token
            prev_pos = cur_pos
            generate_until = cur_pos + 1

            generated = self.tokenizer.decode(tokens[start_pos:generate_until].tolist())
            for stop_symbol in additional_stop_symbols:
                stop_pos = generated.find(stop_symbol)
                if stop_pos != -1:
                    generated = generated[:stop_pos]
                    yield {"text": generated, "end_of_content": True}
                    return

            yield {"text": generated, "end_of_content": False}

        generated = self.tokenizer.decode(tokens[start_pos:generate_until].tolist())
        yield {"text": generated, "end_of_content": True}

    def sample_top_p(self, probs, p):
        """
        Sample a token based on the provided probability distribution using top-p sampling.

        :param probs: The probability distribution for the next token.
        :param p: The cumulative probability threshold for top-p sampling.
        :return: The sampled next token.
        """
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

    def get_quant_blocklist(self) -> List[str]:
        if hasattr(self.llma, "get_quant_blocklist"):
            return ["llma." + x for x in self.llma.get_quant_blocklist()]
        return []
