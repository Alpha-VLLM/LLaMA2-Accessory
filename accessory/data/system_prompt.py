from typing import Dict

def format_prompt(format_dict: Dict, sys_name="alpaca"):
    if sys_name == "alpaca":
        prompt_dict = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
        if "input" not in format_dict or format_dict["input"] is None or format_dict["input"] == "" or format_dict["input"].isspace():
            return prompt_dict['prompt_no_input'].format_map(format_dict)
        else:
            return prompt_dict["prompt_input"].format_map(format_dict)

    elif sys_name == "shortqa":
        prompt =  (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request using a single word or phrase.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
        return prompt.format_map(format_dict)

    elif sys_name == "qg":  # question_generation
        prompt = (
            "Generate a question whose answer is:\n{instruction}\n\n"
            "Question:\n"
        )
        return prompt.format_map(format_dict)

    elif sys_name == "caption":
        return ""

    elif sys_name == "None":
        return "{instruction}".format_map(format_dict)

    else:
        ValueError(sys_name)