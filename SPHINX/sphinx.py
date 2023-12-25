from typing import List, Optional
import torch
import numpy as np
from PIL import Image
from accessory.model.meta import MetaModel

from accessory.data.transform import get_transform
from accessory.data.conversation import default_conversation

class SPHINXModel(MetaModel):
    def generate_response(self, qas: List[List[str]], image: Optional[Image.Image],
                         max_gen_len=512, temperature=0.1, top_p=0.5, seed=0) -> str:
        """

        Args:
            qas: A list of question answer pairs in the form of `[[q1, a1], [q2,a2], ... , [qn, None]]`.
                last answer should be None for generation.
            image: PIL Image for multi-modal understanding
            max_gen_len: generation hyper-param
            temperature: generation hyper-param
            top_p: generation hyper-param
            seed: random seed

        Returns:
            str: response
        """
        # to avoid sampling inconsistency among model parallel workers
        torch.manual_seed(seed)
        np.random.seed(seed)

        if image is not None:
            image = image.convert("RGB")
            target_size = getattr(self.llma, 'image_size', 224)  # 448 for SPHINX-1k, 224 for SPHINX
            transform = get_transform("padded_resize", target_size)
            image = transform(image).to(list(self.parameters())[0])

        conv = default_conversation()
        assert qas[-1][1] is None

        conv.load_qas(qas)
        prompt = conv.get_prompt()
        # print(prompt)

        # each turn of response ends with `conv_seq`
        conv_sep = conv.response_end_signal

        # since MetaModel.generate is originally designed for batched inference,
        # we need to form a batch of size 1 here
        response = self.generate(
            prompts=[prompt],
            images=image.unsqueeze(0) if image is not None else None,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            additional_stop_symbols=[conv_sep]
        )[0]

        return response
