# Language-only Full-Parameter Finetuning

## Single-turn LLaMA2-7B on Alpaca

**Script:**

+ {link2repo}`[exps/finetune/sg/alpaca.sh](accessory/exps/finetune/sg/alpaca.sh)`

**Data:**

+ [💾alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/alpaca/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/alpaca_finetuned
```

**Example:**

```{card}
*Instruction:*

Write an introduction for the github repository "LLaMA2-Accesory: An Open-source Toolkit for LLM Development".

*Answer:*

Hello and welcome to the LLaMA2-Accesory GitHub repository! This is an open-source toolkit designed specifically for the development of LLM (Language Modeling) applications. Built on top of the powerful LLaMA2 library, this toolkit provides a user-friendly interface and a set of robust utilities to help you build and train your own LLM models with ease. Whether you are a seasoned AI developer or just getting started with LLaMA2, this toolkit has got you covered with its robust features and extensive documentation. Let's get started on building your LLaMA2-based application today!

[*More examples here*](../examples/finetune/sg/alpaca.md)
```

---

## Single-turn InternLM-7B on Alpaca

**Script:**

+ {link2repo}`[exps/finetune/sg/alpaca_internLM.sh](accessory/exps/finetune/sg/alpaca_internLM.sh)`

**Data:**

+ [💾alpaca_gpt4_data_zh.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json)

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/alpaca_internLM_zh/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/alpaca_finetuned
```

**Example:**

```{card}
**Human:** 生成一篇关于人工智能未来发展趋势的文章

**Assistant:** 人工智能的未来发展趋势

人工智能（AI）技术在过去几十年中取得了长足的进步，它正在改变我们的生活方式和商业运作方式。未来，人工智能将如何发展？本文将探讨人工智能的未来发展趋势。

首先，人工智能将更加智能化。随着机器学习和深度学习技术的不断发展，人工智能将能够更好地理解和处理复杂数据，从而做出更加智能的决策。例如，在医疗领域，人工智能将能够更好地诊断疾病，为患者提供更精准的治疗方案。在金融领域，人工智能将能够更好地预测市场走势，为投资者提供更精准的投资建议。

其次，人工智能将更加人性化。随着自然语言处理技术的不断发展，人工智能将能够更好地与人类进行交流和沟通。例如，智能语音助手将能够更好地理解人类的语言，为人类提供更加人性化的服务。此外，人工智能也将更加注重隐私保护，为人类提供更加安全、可靠的个人信息保护措施。

最后，人工智能将更加普及。随着人工智能技术的不断发展，它将逐渐渗透到人们的日常生活中，为人们提供更加便捷、高效的智能化服务。例如，智能家居系统将能够更好地控制家庭中的各种设备，为人们提供更加舒适、便捷的生活环境。

[*More examples here*](../examples/finetune/sg/intern_alpaca_zh.md)
```

---

## Single-turn LLaMA2-7B on Gorilla

**Script:**

+ {link2repo}`[exps/finetune/sg/gorilla.sh](accessory/exps/finetune/sg/gorilla.sh)`

**Data:**

+ Here we take tensorflowhub as example, other subset can be handled in similar way:
+ Download data from [💾here](https://github.com/ShishirPatil/gorilla/blob/main/data/apibench/tensorflow_train.json)
+ Run {link2repo}`[tools/data_conversion/to_alpaca/gorilla.py](accessory/tools/data_conversion/to_alpaca/gorilla.py)` with specified `--data_path` to reformat original data
+ Make sure {link2repo}`[gorilla.yaml](accessory/configs/data/finetune/sg/gorilla.yaml)` is correctly specified to formatted data file

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/gorilla/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
torchrun --nproc-per-node=1  demos/single_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/gorilla_finetuned
```

**Accuracies:**

Tuning scripts used here can be found in:  {link2repo}`[Zero-init Attenion](accessory/exps/finetune/sg/gorilla_llamaAdapter.sh)`(i.e. LLaMA-adapter),  {link2repo}`[Bias-norm Tuning](accessory/exps/finetune/sg/gorilla_llamaPeft_normBias.sh)`,  {link2repo}`[LoRA + Bias-norm](accessory/exps/finetune/sg/gorilla_llamaPeft_normBiasLora.sh)`.

| Methods (TensorFlow Hub) | Overall ↑ | Hallu ↓ |
| ------------------------ | --------- | ------- |
| Official                 | 83.79     | 5.40    |
| Full finetune            | 88.84     | 0.14    |
| Zero-init Attention      | 84.97     | 1.04    |
| Bias-norm Tuning         | 80.36     | 7.29    |
| LoRA + Bias-norm         | 87.95     | 2.38    |


**Example:**

:::{card}
*Instruction:*


I am working on a project where I need to cluster similar images of street art. How can I get the relevant feature vectors from these images for clustering?

*Answer:*

<<\<domain\>>>: Image feature vector,

<<\<api_call\>>>: hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'),

<<\<api_provider\>>>: TensorFlow Hub,

<<\<explanation\>>>: This API call loads a pretrained MobileNetV2 model from TensorFlow Hub, which has been trained on the ImageNet dataset. This model is used for extracting feature vectors from images. You can feed your street art images into this model to obtain feature vectors, which can then be used for clustering.

<<\<code\>>>:

```
import tensorflow as tf
import tensorflow_hub as hub
model = tf.keras.Sequential([
hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4',
               input_shape=(224, 224, 3),
               trainable=False)
])

image = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

features = model.predict(image_array)
```

[*More examples here*](../examples/finetune/sg/gorilla.md)
:::

---
## Multi-turn LLaMA2-7B on ShareGPT

**Script:**

+ {link2repo}`[exps/finetune/sg/dialog_sharegpt.sh](accessory/exps/finetune/sg/dialog_sharegpt.sh)`

**Data:**

+ Please collect and process the data on your own. {link2repo}`[Here](data_example/ShareGPT.json)` is a toy example showing the proper format of the data file.

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/dialog_sharegpt/consolidated.00-of-01.model.pth)

**Benchmark on OpenCompass:**
| Model | Overall | Exam | Language | Knowledge | Reasoning | Understanding |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA-2-7B | 41.68 | 36.43 | 50.7 | 48.46 | 41.05 | 37.47 |
| LLaMA-2-7B-ShareGPT | 43.00 | 42.43 | 51.32 | 47.85 | 42.57 | 37.35 |

> Please check [OpenCompass](https://github.com/InternLM/opencompass) for more details.

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/sharegpt_finetuned
```

**Example:**
:::{card}
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

**Human:** Assuming you are a native English speaker, please rewrite the content I provided to make it more concise and professional. Do not change the willingness of the sentence. Don't reply with anything other than the revised sentence.

My content is:

```
I want study math everyday because I like it. It makes me feel very good. I do not like English because it is hard. I can not remember the words
```

**Assistant:** "I am eager to study mathematics daily as I find it fascinating. It brings me great satisfaction. On the other hand, I do not have the same enthusiasm for English as it is a challenging subject for me. Memorizing vocabulary and grammar rules is not my strong suit."

**Human:** My second content is:

```
My name is Jimmy, a master student studing computer science. I have done some projects. For example, I designed a image classification model, its name is CLSModel. I learned a lot from this project.
```

**Assistant:** "My name is Jimmy, and I am a master's student pursuing a degree in computer science. I have had the opportunity to work on various projects, one of which was an image classification model called CLSModel. Through this project, I gained valuable experience and knowledge that has helped me grow as a computer scientist."

[*More examples here*](../examples/finetune/sg/dialog_sharegpt.md)
:::
---
## Multi-turn LLaMA2-70B on ShareGPT

**Script:**

+ {link2repo}`[exps/finetune/sg/dialog_sharegpt_70B.sh](accessory/exps/finetune/sg/dialog_sharegpt_70B.sh)`

**Data:**

+ Please collect and process the data on your own. {link2repo}`[Here](../data_example/ShareGPT.json)` is a toy example showing the proper format of the data file.

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/sg/dialog_sharegpt_70b)

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/sharegpt_finetuned --n_gpus 4
```

**Example:**
:::{card}
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

**Human:** Describe a room with three switches, controlling three lamps in another room. You can only enter the controlled room once. How can you determine which switch controls each lamp?

**Assistant:** To determine which switch controls each lamp, you can use a process of elimination. Here's one possible approach:

1. Enter the controlled room and take note of the positions of the lamps. Are they all on or off?
2. If all lamps are off, turn on switch 1 and enter the controlled room again. If one of the lamps turns on, you know that switch 1 controls that lamp.
3. If the lamp does not turn on, turn off switch 1 and repeat the process with switch 2. If a lamp turns on, you know that switch 2 controls that lamp.
4. If no lamp turns on with switch 2, turn off switch 2 and repeat the process with switch 3. If a lamp turns on, you know that switch 3 controls that lamp.
5. If no lamp turns on with any of the switches, it's possible that there is a problem with the wiring or the lamps themselves.

By following this process, you should be able to determine which switch controls each lamp.

[*More examples here*](../examples/finetune/sg/dialog_sharegpt_70b.md)
:::
---
## Multi-turn LLaMA2-7B on LIMA

**Script:**

+ {link2repo}`[exps/finetune/sg/dialog_lima.sh](accessory/exps/finetune/sg/dialog_lima.sh)`

**Data:**

+ [💾lima_vicuna_format.json](https://huggingface.co/datasets/64bits/lima_vicuna_format/raw/main/lima_vicuna_format.json)

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/dialog_lima/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/lima_finetuned
```

**Example:**
:::{card}
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

**Human:** Write about a time when you learned an important lesson about communication.

**Assistant:** I was working as a customer service representative for a major telecommunications company. It was a pretty typical day, lots of customers with different issues.
One particular call stuck with me though. An elderly gentleman was having trouble with his internet. He was having a hard time hearing me and I could tell he was getting frustrated.
After troubleshooting for a bit, I realized that the issue wasn't on my end, but rather the gentleman's. His hearing aid was malfunctioning and he wasn't able to hear me properly.
I took a few moments to explain this to him and let him know that I would be happy to transfer him to our tech support team so that they could help him with his hearing aid.
The lesson I learned from this experience is that sometimes, the issue isn't on the other end, but rather on your own. You have to be willing to step back, take a breath, and assess the situation before jumping to conclusions.

**Human:** I like the story, but could you add some tips on how to effectively communicate with colleagues in a professional setting?

**Assistant:** Sure, here are some tips on how to effectively communicate with colleagues in a professional setting:

* Be clear and concise in your communication. Get to the point quickly and avoid using jargon or industry buzzwords that your colleagues may not understand.
* Listen actively when others are speaking. Nod your head, ask clarifying questions, and avoid interrupting.
* Respect the time and attention of your colleagues. Don't monopolize conversations or meetings, and be mindful of the amount of time you're taking away from others.
* Be open and honest in your communication. Don't withhold information or try to manipulate others.
* Follow up on communication. Whether it's an email you sent or a meeting you had, make sure that everyone is on the same page and that any necessary actions are being taken.

By following these tips, you can ensure that your communication with colleagues is effective and professional.

**Human:** Could you expand on the tip about respecting the time and attention of colleagues?

**Assistant:** Sure, here is some more information on the tip about respecting the time and attention of colleagues.

* Don't monopolize conversations or meetings. If someone else has something to contribute, let them. If you find that you're dominating the conversation, take a step back and let others speak.
* Be mindful of the amount of time you're taking away from others. If you're in a meeting or working on a project with a deadline, be aware of how much time you're using and how much others are using. Don't take up more time than necessary.
* Avoid multitasking. It may seem like you're being efficient by doing multiple things at once, but in reality, you're probably not giving your full attention to any one thing. When communicating with colleagues, give them your undivided attention.
* Be considerate of others' schedules. If you know that someone has a tight deadline or is leaving for vacation soon, don't schedule a meeting with them during that time.

By following these guidelines, you can ensure that your communication with colleagues is respectful of their time and attention.

[*More examples here*](../examples/finetune/sg/dialog_lima.md)
:::
---
## Single-turn LLaMA2-7B on WizardLM

**Script:**

+ {link2repo}`[exps/finetune/sg/dialog_wizardLM.sh](accessory/exps/finetune/sg/dialog_wizardLM.sh)`

**Data:**

+ [💾WizardLM](https://github.com/nlpxucan/WizardLM)

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/dialog_wizardLM/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/wizardLM_finetuned
```

**Example:**
:::{card}

A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

**Human:** Write a quicksort code. You need to use C++ as the programming language and generate a function called quicksort.

**Assistant:** Here's a quicksort implementation in C++:

```c++
#include <iostream>
#include <algorithm>

using namespace std;

void quicksort(int arr[], int left, int right) {
int i = left, j = right;
int pivot = arr[(left + right) / 2];

while (i <= j) {
  while (arr[i] < pivot) {
      i++;
  }
  while (arr[j] > pivot) {
      j--;
  }
  if (i <= j) {
      swap(arr[i], arr[j]);
      i++;
      j--;
  }
}

quicksort(arr, left, j);
quicksort(arr, i, right);
}

int main() {
int arr[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
int n = sizeof(arr) / sizeof(arr[0]);

cout << "Original array: ";
for (int i = 0; i < n; i++) {
  cout << arr[i] << " ";
}
cout << endl;

quicksort(arr, 0, n - 1);

cout << "Sorted array: ";
for (int i = 0; i < n; i++) {
  cout << arr[i] << " ";
}
cout << endl;

return 0;
}
```

The quicksort function takes in an array arr, as well as the left and right indices of the subarray to be sorted. It first finds a pivot element, which is the middle element of the subarray. It then rearranges the elements in the subarray such that all elements less than the pivot are to its left, and all elements greater than the pivot are to its right. It recursively calls itself on the left and right subarrays until the entire array is sorted.

In the main function, we define an unsorted array and call quicksort on it. We then print out the sorted array to verify that the sorting was successful.

 [*More examples here*](../examples/finetune/sg/dialog_wizardLM.md)
:::

---
## Single-turn CodeLLaMA-7B on WizardCode

**Script:**

+ {link2repo}`[exps/finetune/sg/dialog_wizardCode.sh](accessory/exps/finetune/sg/dialog_wizardCode.sh)`

**Data:**

+ [💾evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)

**Model Release:**

+ [🤗checkpoint](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/blob/main/finetune/sg/dialog_wizardcode_codellama/consolidated.00-of-01.model.pth)

**Host Local Demo:**

```bash
python demos/multi_turn.py \
--llama_config /path/to/params.json --tokenizer_path /path/to/tokenizer.model \
--pretrained_path /path/to/wizardCode_finetuned
```

:::{attention}

The Code LLaMA series has different `params.json` and `tokenizer.model` files from LLaMA and LLaMA2. Make sure to use the Code-LLaMA version of these files.

:::

---

*More use cases coming soon...*
