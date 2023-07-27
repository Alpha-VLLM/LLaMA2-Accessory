# Fine-tuning

This document demonstrates the fine-tuning use cases supported by LLaMA2-Accessory

## Contents
* [Prerequisites](#prerequisites)
* [How to Apply Delta Weights](#how-to-apply-delta-weights)
* [Full-Parameter Fine-tuning](#full-parameter-fine-tuning)
  - [Single-turn instruction-tuning of LLaMA2-7B on alpaca](#single-turn-instruction-tuning-of-llama2-7b-on-alpaca)
  - [Multi-turn instruction-tuning of LLaMA2-7B on ShareGPT](#multi-turn-instruction-tuning-of-llama2-7b-on-sharegpt)
  - [Multi-turn instruction-tuning of LLaMA2-7B-Chat on ShareGPT](#multi-turn-instruction-tuning-of-llama2-7b-chat-on-sharegpt)
  - [Multi-turn instruction-tuning of LLaMA2-7B on LIMA](#multi-turn-instruction-tuning-of-llama2-7b-on-lima)
  - [Single-turn instruction-tuning of LLaMA2-7B on gorilla](#single-turn-instruction-tuning-of-llama2-7b-on-gorilla)

> ## Prerequisites
>
> To run our provided experiment scripts on you own machine, please first adjust the following configurations:
>
> + Modify the value of the `pretrained_path` variable in the `.sh` file. This variable should point to the directory containing checkpoints to fine-tune from.
>   + If you fine-tune from the officianl LLaMA / LLaMA2 checkpoints released by META, the directory should be like:
>     ```
>     pretrained_path
>     ├── consolidated.00.pth
>     ├── consolidated.01.pth
>     └── ...
>     ```
>
>     and your should set `pretrained_type=meta_ori` in the `.sh` file.
>   + Alternatively, you may also fine-tune from checkpoints saved by LLaMA2-Accessory. In such cases, the directory should be like:
>     ```
>     pretrained_path
>     ├── consolidated.00-of-**.model.pth
>     ├── consolidated.01-of-**.model.pth
>     └── ...
>     ```
>
>   and your should set `pretrained_type=consolidated` in the `.sh` file
> + Point `llama_config` in `.sh` to the model parameter file (usually named as `params.json`) that differentiates model sizes (7B, 13B, ...)
> + Point `tokenizer_path` in `.sh` to the tokenizer file (usually named as `tokenizer.model`)
> + The `data_config` argument in the `.sh` file points to a `.yaml` file defining the fine-tuning datasets. You need to first download the data and modify the paths in `.yaml` to the correct location

> ## How to Apply Delta Weights
>
> We release checkpoints as delta weights to comply with the LLaMA2 model license. You can add our delta to the original LLaMA2 weights to obtain the finetuned weights. Instructions:
>
> 1. After agreeing to the License, Acceptable Use Policy, and Meta's privacy policy, proceed to download the LLaMA2 weights from [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
> 2. Utilize the following scripts to obtain finetuned weights by applying our delta. Make sure to download the delta weights from the [model release page](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory).
>    ```bash
>    # For Merging
>    python tools/weight_operate.py  --pretrained_path /path/to/llama2/consolidated.00.pth --delta_path /path/to/delta.pth --output_path /path/to/finetuned.pth --operate_type delta/peft
>    # For Separation
>    python tools/weight_operate.py  --pretrained_path /path/to/llama2/consolidated.00.pth --delta_path /path/to/finetuned.pth --output_path /path/to/delta.pth --operate_type extract
>    ```

## Full-Parameter Fine-tuning

### Single-turn instruction-tuning of LLaMA2-7B on alpaca

**Script:**

+ [exps/finetune/sg/alpaca.sh](../accessory/exps/finetune/sg/alpaca.sh)

**Data:**

+ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json

**Model Release:**

+ [checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/alpaca/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/alpaca_finetuned
```

**Example:**

> *Instruction:*
>
> Write an introduction for the github repository "LLaMA2-Accesory: An Open-source Toolkit for LLM Development".
>
> *Answer:*
>
> Hello and welcome to the LLaMA2-Accesory GitHub repository! This is an open-source toolkit designed specifically for the development of LLM (Language Modeling) applications. Built on top of the powerful LLaMA2 library, this toolkit provides a user-friendly interface and a set of robust utilities to help you build and train your own LLM models with ease. Whether you are a seasoned AI developer or just getting started with LLaMA2, this toolkit has got you covered with its robust features and extensive documentation. Let's get started on building your LLaMA2-based application today!
>
> [*More examples here*](./examples/finetune/sg/alpaca.md)

---

### Multi-turn instruction-tuning of LLaMA2-7B on ShareGPT

**Script:**

+ [exps/finetune/sg/dialog_sharegpt.sh](../accessory/exps/finetune/sg/dialog_sharegpt.sh)

**Data:**

+ Please collect and process the data on your own. [Here](../data_example/ShareGPT.json) is a toy example showing the proper format of the data file.

**Model Release:**

+ [checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/dialog_sharegpt/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/sharegpt_finetuned
```

**Example:**

> A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
>
> **Human:** Assuming you are a native English speaker, please rewrite the content I provided to make it more concise and professional. Do not change the willingness of the sentence. Don't reply with anything other than the revised sentence.
>
> My content is:
>
> ```
> I want study math everyday because I like it. It makes me feel very good. I do not like English because it is hard. I can not remember the words
> ```
>
> **Assistant:** "I am eager to study mathematics daily as I find it fascinating. It brings me great satisfaction. On the other hand, I do not have the same enthusiasm for English as it is a challenging subject for me. Memorizing vocabulary and grammar rules is not my strong suit."
>
> **Human:** My second content is:
>
> ```
> My name is Jimmy, a master student studing computer science. I have done some projects. For example, I designed a image classification model, its name is CLSModel. I learned a lot from this project.
> ```
>
> **Assistant:** "My name is Jimmy, and I am a master's student pursuing a degree in computer science. I have had the opportunity to work on various projects, one of which was an image classification model called CLSModel. Through this project, I gained valuable experience and knowledge that has helped me grow as a computer scientist."
>
> [*More examples here*](./examples/finetune/sg/dialog_sharegpt.md)

---

### Multi-turn instruction-tuning of LLaMA2-7B-Chat on ShareGPT

**Script:**

+ [exps/finetune/sg/llama2_chat_sharegpt.sh](../accessory/exps/finetune/sg/llama2_chat_sharegpt.sh)

**Data:**

+ Please collect and process the data on your own. [Here](../data_example/ShareGPT.json) is a toy example showing the proper format of the data file.

**Model Release:**

+ [checkpoint (TODO)]()

**Host Local Demo:**

```bash
python demos/llama2_chat.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/llama2_chat_sharegpt
```

**Example:**

> **Human:** Thanks for your work, it's pretty pretty helpful.
> My conda environment: `mmcv 0.2.16`, `cuda 11.1`, `torch 1.8.0`, `RTX 3080`.
> 
> When in config.py use `roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2)`, I can only use two gpu, when use four, then print an error, "THCudaCheck FAIL file=ReDet/mmdet/ops/roi_align/src/roi_ane=292 error=700 : an illegal memory access was encountered." But it's ok for use `roi_layer=dict(type='RoIPool', out_size=7)` to fully use 4 gpu it's so weird.
>
> Therefore I am sure there is a bug left in `roi_align_kernel.cu`, I am debugging it out now.
> Any idea? thx
>
> **Assistant:** It's possible that there is a bug in the `roi_align_kernel.cu` file that is causing the issue when using four GPUs. It's also possible that there is a compatibility issue between the versions of the libraries you are using.
>
> One thing you can try is to check the memory layout of the input data and the kernel parameters to see if there is any mismatch that could cause the illegal memory access error.
>
> Another thing you can try is to use a different version of the `torch` library that is compatible with your CUDA version (version 11.1) and see if the issue persists.
>
> You can also try to use `torch.cuda.device` to specify the GPU you want to use, instead of relying on the default behavior of the `torch.cuda.current_device()` function.
> 
> It's also possible that the issue is not related to the code you provided and it could be related to other parts of the code or the environment you are running it on.
>
> I hope this helps, let me know if you have any other questions.
>
> **Human:** Can you give me some suggestions on how to debug CUDA error?
>
> **Assistant:** Sure, here are some general suggestions for debugging CUDA errors:
>
> * Check the error message: The error message usually contains information about the specific error that occurred and can often help you determine the root cause of the problem.
> * Use the CUDA debugger: The CUDA debugger is a tool that allows you to step through your code and inspect the state of the GPU and memory at each step. It can help you identify memory access errors and other issues.
> * Check the memory layout: Make sure that the memory layout of your data is compatible with the kernel parameters. For example, if the kernel expects the data to be aligned to a certain boundary, make sure that your data is aligned accordingly.
> * Use profiling tools: Profiling tools like nvprof can help you identify performance bottlenecks and memory usage in your code.
> * ...
>
> These are just a few general suggestions for debugging CUDA errors. The specific steps you need to take will depend on the specific error message and the code you are working with.
> 
> [*More examples here*](./examples/finetune/sg/llama2-chat.md)

---

### Multi-turn instruction-tuning of LLaMA2-7B on LIMA

**Script:**

+ [exps/finetune/sg/dialog_lima.sh](../accessory/exps/finetune/sg/dialog_lima.sh)

**Data:**

+ https://huggingface.co/datasets/64bits/lima_vicuna_format/raw/main/lima_vicuna_format.json

**Model Release:**

+ [checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/dialog_lima/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/lima_finetuned
```

**Example:**

> A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
>
> **Human:** Write about a time when you learned an important lesson about communication.
>
> **Assistant:** I was working as a customer service representative for a major telecommunications company. It was a pretty typical day, lots of customers with different issues.
> One particular call stuck with me though. An elderly gentleman was having trouble with his internet. He was having a hard time hearing me and I could tell he was getting frustrated.
> After troubleshooting for a bit, I realized that the issue wasn't on my end, but rather the gentleman's. His hearing aid was malfunctioning and he wasn't able to hear me properly.
> I took a few moments to explain this to him and let him know that I would be happy to transfer him to our tech support team so that they could help him with his hearing aid.
> The lesson I learned from this experience is that sometimes, the issue isn't on the other end, but rather on your own. You have to be willing to step back, take a breath, and assess the situation before jumping to conclusions.
>
> **Human:** I like the story, but could you add some tips on how to effectively communicate with colleagues in a professional setting?
>
> **Assistant:** Sure, here are some tips on how to effectively communicate with colleagues in a professional setting:
>
> * Be clear and concise in your communication. Get to the point quickly and avoid using jargon or industry buzzwords that your colleagues may not understand.
> * Listen actively when others are speaking. Nod your head, ask clarifying questions, and avoid interrupting.
> * Respect the time and attention of your colleagues. Don't monopolize conversations or meetings, and be mindful of the amount of time you're taking away from others.
> * Be open and honest in your communication. Don't withhold information or try to manipulate others.
> * Follow up on communication. Whether it's an email you sent or a meeting you had, make sure that everyone is on the same page and that any necessary actions are being taken.
>
> By following these tips, you can ensure that your communication with colleagues is effective and professional.
>
> **Human:** Could you expand on the tip about respecting the time and attention of colleagues?
>
> **Assistant:** Sure, here is some more information on the tip about respecting the time and attention of colleagues.
>
> * Don't monopolize conversations or meetings. If someone else has something to contribute, let them. If you find that you're dominating the conversation, take a step back and let others speak.
> * Be mindful of the amount of time you're taking away from others. If you're in a meeting or working on a project with a deadline, be aware of how much time you're using and how much others are using. Don't take up more time than necessary.
> * Avoid multitasking. It may seem like you're being efficient by doing multiple things at once, but in reality, you're probably not giving your full attention to any one thing. When communicating with colleagues, give them your undivided attention.
> * Be considerate of others' schedules. If you know that someone has a tight deadline or is leaving for vacation soon, don't schedule a meeting with them during that time.
>
> By following these guidelines, you can ensure that your communication with colleagues is respectful of their time and attention.
>
> [*More examples here*](./examples/finetune/sg/dialog_lima.md)

---

### Single-turn instruction-tuning of LLaMA2-7B on gorilla

**Script:**

+ [exps/finetune/sg/dialog_gorilla.sh](../accessory/exps/finetune/sg/gorilla.sh)

**Data:**

+ https://github.com/ShishirPatil/gorilla/blob/main/data/apibench/tensorflow_train.json
+ run [tools/data_conversion/to_alpaca/gorilla.py](../accessory/tools/data_conversion/to_alpaca/gorilla.py) to reformat original data.

**Model Release:**

+ [checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/gorilla/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/gorilla_finetuned
```

**Example:**

> *Instruction:*
>
> I am working on a project where I need to cluster similar images of street art. How can I get the relevant feature vectors from these images for clustering?
>
> *Answer:*
>
> <<\<domain\>>>: Image feature vector,
>
> <<\<api_call\>>>: hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'),
>
> <<\<api_provider\>>>: TensorFlow Hub,
>
> <<\<explanation\>>>: This API call loads a pre-trained MobileNetV2 model from TensorFlow Hub, which has been trained on the ImageNet dataset. This model is used for extracting feature vectors from images. You can feed your street art images into this model to obtain feature vectors, which can then be used for clustering.
>
> <<\<code\>>>:
>
> ```
> import tensorflow as tf
> import tensorflow_hub as hub
> model = tf.keras.Sequential([
>     hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4',
>                    input_shape=(224, 224, 3),
>                    trainable=False)
> ])
>
> image = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
> image_array = tf.keras.preprocessing.image.img_to_array(image)
> image_array = tf.expand_dims(image_array, 0)
>
> features = model.predict(image_array)
> ```
>
> [*More examples here*](./examples/finetune/sg/gorilla.md)

---

*More use cases coming soon...*
