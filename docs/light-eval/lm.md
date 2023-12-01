# Language Model evaluation
## environment settings
Before running the Light-eval, users must ensure that they have correctly installed and configured all necessary environments according to the instructions in the [Installation Document](../install.md).

## BIG-Bench-Hard
### Prerequisites
**dataset**
```
data/BIG-Bench-Hard/
├── bbh
│   ├── boolean_expressions.json
│   ├── causal_judgement.json
│   └── ...
└── ...
```
The dataset is available at [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)

```
cd data/
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git
```

### evaluating
```bash
sh scripts/run_bbh.sh
```
**Script Demo**
```
task=bbh
pretrained_type=meta_ori
pretrained_path=/path/to/your/model_dir
llama_config=/path/to/your/config
tokenizer_path=/path/to/your/tokenizer
data_dir='data/BIG-Bench-Hard'

nproc_per_node=1
model_parallel=1
master_port=23456

exp_name=your/model/name
mkdir -p logs/"$exp_name"

torchrun --nproc-per-node="$nproc_per_node" --master_port "$master_port" src/eval_"$task".py \
    --pretrained_type "$pretrained_type" \
    --llama_config "$llama_config" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_path "$pretrained_path" \
    --data_dir "$data_dir" \
    2>&1 | tee logs/"$exp_name"/"$task".log
```

- `task` : variable used to determine the result file name and log name, set by default to the name of the benchmark.
- `exp_name` :  variable used to determine the result file name and log name, set by default to the name of the model.
- `llama_config` : variable should point to the `params.json` file.
- `tokenizer_path` : variable  should point to the `tokenizer.model` file.

- `pretrained_path` variable in the to the directory containing checkpoints.

- `pretrained_type` :
  - For the official LLaMA / LLaMA2 checkpoints released by META, you should set `pretrained_type=meta_ori`.
  - For  the checkpoints finetuned / saved by LLaMA2-Accessory, you should set `pretrained_type=consolidated`.
- `data_dir` : Please note that the dataset is stored according to the storage structure described in **dataset**, and you need to point the variable to the dataset folder
- `nproc_per_node` , `model_parallel ` :  variables set according to the model.
- `master_port` : variable that set the port used by `torchrun`.



## MMLU
### Prerequisites

**dataset**

```
data/mmlu/
└── data
    ├── dev
    │   ├── abstract_algebra_dev.csv
    │   ├── anatomy_dev.csv
    │	└── ...
    ├── val
    ├── test
    └── ...
```

The dataset is available for download [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar).

```
mkdir data/mmlu
cd data/mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
```

### evaluating

```bash
sh scripts/run_mmlu.sh
```



## Math

### Prerequisites

**dataset**

```
data/math/
└── MATH_test.jsonl
```
The dataset is ready in the {link2repo}`[light-eval/data/math/](light-eval/data/math/)`

### evaluating

```bash
sh scripts/run_math.sh
```



## GSM8K

### Prerequisites

**dataset**

```
data/gsm8k/
└── gsm8k_test.jsonl
```
The dataset is ready in the {link2repo}`[light-eval/data/gsm8k/](light-eval/data/gsm8k/)`

### evaluating

```bash
sh scripts/run_gsm8k.sh
```



## HumanEval

### Prerequisites

**dataset**

```
data/human-eval/
├── data
│   ├── example_problem.jsonl
│   ├── example_samples.jsonl
│   └── HumanEval.jsonl.gz
└──...
```

The dataset is available at [openai/human-eval](https://github.com/openai/human-eval)

```
cd data/
git clone https://github.com/openai/human-eval.git
pip install -e human-eval
```

### evaluating

```bash
sh scripts/run_humaneval.sh
```



## CEVAL

### Prerequisites

**dataset**

```
data/ceval/
├── dev
│   ├── accountant_dev.csv
│   └── ...
├── test
└── val
```

The dataset is available at [🤗Hugging Face/ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)

```
mkdir data/ceval
cd data/ceval
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip 
```

### evaluating

```bash
sh scripts/run_ceval.sh
```



## CMMLU

### Prerequisites

**dataset**

```
data/cmmlu/
├── dev
│   ├── agronomy.csv
│   └── ...
└── test
```

The dataset is available at [🤗Hugging Face/haonan-li/cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)

```
mkdir data/cmmlu
cd data/cmmlu
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
unzip cmmlu_v1_0_1.zip
```

### evaluating

```bash
sh scripts/run_cmmlu.sh
```
