import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + '\n\n' + self.sep
            for role, message in self.messages:
                if message is not None:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + '\n' + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)



conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Give three tips for staying healthy."),
        ("Assistant",
         "Sure, here are three tips for staying healthy:\n"
         "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
         "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
         "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
         "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
         "activities at least two days per week.\n"
         "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
         "vegetables, whole grains, lean proteins, and healthy fats can help support "
         "your overall health. Try to limit your intake of processed and high-sugar foods, "
         "and aim to drink plenty of water throughout the day.\n"
         "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
         "and mental health. Adults should aim for seven to nine hours of sleep per night. "
         "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
         "help improve the quality of your sleep.")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),

    # (
    #     ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
    #     ("Assistant",
    #         "Renewable energy sources are those that can be replenished naturally in a relatively "
    #         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
    #         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
    #         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
    #         "renewable and non-renewable energy sources:\n"
    #         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
    #         "energy sources are finite and will eventually run out.\n"
    #         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
    #         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
    #         "and other negative effects.\n"
    #         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
    #         "have lower operational costs than non-renewable sources.\n"
    #         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
    #         "locations than non-renewable sources.\n"
    #         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
    #         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
    #         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
    #         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    # )
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
- You are a helpful language and vision assistant.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mpt_text = Conversation(
    system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

simple_conv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there! How can I help you today?")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_multimodal = Conversation(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
           "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_mpt_multimodal = Conversation(
    system="""<|im_start|>system
- You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

simple_conv_legacy = Conversation(
    system="You are LLaVA, a large language model trained by UW Madison WAIV Lab."
           "You are designed to assist human with a variety of tasks using natural language."
           "Follow the instructions carefully.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!\n\n### Response:"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1 = Conversation(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
           "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_v1_2
conv_templates = {
    "default": conv_v1_2,
    "simple": simple_conv,
    "simple_legacy": simple_conv_legacy,
    "multimodal": simple_conv_multimodal,
    "mpt_multimodal": simple_conv_mpt_multimodal,
    "llava_v1": conv_llava_v1,

    # fastchat
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "mpt": conv_mpt,
    "mpt_text": conv_mpt_text,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
