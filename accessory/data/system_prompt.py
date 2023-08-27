first = [True]

def format_prompt(instruction, input=None, sys_name="alpaca"):
    if input is None or input == "" or input.isspace():
        input = None
    if sys_name == "alpaca":
        PROMPT_DICT = {
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
        if input is None or input=='':
            return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
        else:
            return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})

    elif sys_name == "qg":  # question_generation
        prompt = (
            "Generate a question whose answer is:\n{instruction}\n\n"
            "Question:\n"
        )
        if first[0]:
            print(prompt.format_map({'instruction': instruction}))
            first[0] = False
        return prompt.format_map({'instruction': instruction})