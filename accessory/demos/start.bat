@echo off
setlocal enabledelayedexpansion

:download
echo Do you want to download the model? (y/n):
set /p DOWNLOAD=
if "%DOWNLOAD%"=="y" (
    echo Downloading the model...
    python tools\download.py
    goto :scenario
) else if "%DOWNLOAD%"=="n" (
    goto :scenario
) else (
    goto :download
)

:scenario
echo Choose the inference scenario:
echo 1) Single-turn Dialogue (Single-modal)
echo 2) Multi-turn Dialogue (Single-modal)
echo 3) Multi-modal Dialogue (Multi-modal)
set /p SCENARIO=
if "%SCENARIO%"=="1" (
    set OPTIONS=internlm llama_adapter llama_peft llama
    goto :llama_type
) else if "%SCENARIO%"=="2" (
    set OPTIONS=internlm llama_adapter llama_peft llama
    goto :llama_type
) else if "%SCENARIO%"=="3" (
    set OPTIONS=llama_qformerv2 llama_qformerv2_peft llama_ens
    goto :llama_type
) else (
    goto :scenario
)

:llama_type
echo Please input llama_type:
for %%i in (!OPTIONS!) do echo %%i
set /p LLAMA_TYPE=
set QUANT=
echo Do you want to enable Quantization? (y/n):
set /p QUANT=
if "%QUANT%"=="y" (
    set QUANT=--quant
) else (
    set QUANT=
)

echo Enter the model size (1: 7B, 2: 13B, 3: 34B, 4: 70B):
set /p SIZE_CHOICE=
if "%SIZE_CHOICE%"=="1" (
    set NPROC=1
) else if "%SIZE_CHOICE%"=="2" (
    set NPROC=2
) else if "%SIZE_CHOICE%"=="3" (
    set NPROC=4
) else if "%SIZE_CHOICE%"=="4" (
    set NPROC=8
) else (
    set NPROC=1
)

echo Enter the master port (default is 12345):
set /p PORT=
if "%PORT%"=="" (
    set PORT=12345
)

echo Enter the path to params.json:
set /p PARAMS=
echo Enter the path to tokenizer.model:
set /p TOKENIZER=
echo Enter the path to the pretrained model:
set /p PRETRAINED=

if "%LLAMA_TYPE%"=="llama_peft" (
    set PARAMS=!PARAMS! configs\model\finetune\sg\llamaPeft_normBiasLora.json
)

if "%SCENARIO%"=="1" (
    torchrun --nproc_per_node=!NPROC! --master_port=!PORT! demos\single_turn.py --llama_config=!PARAMS! --tokenizer_path=!TOKENIZER! --pretrained_path=!PRETRAINED! !QUANT! --llama_type=!LLAMA_TYPE!
) else if "%SCENARIO%"=="2" (
    python  demos\multi_turn.py  --n_gpus=!NPROC! --master_port=!PORT! --llama_config=!PARAMS! --tokenizer_path=!TOKENIZER! --pretrained_path=!PRETRAINED! !QUANT! --llama_type=!LLAMA_TYPE!
) else if "%SCENARIO%"=="3" (
    torchrun --nproc_per_node=!NPROC! --master_port=!PORT! demos\single_turn_mm.py --llama_config=!PARAMS! --tokenizer_path=!TOKENIZER! --pretrained_path=!PRETRAINED! !QUANT! --llama_type=!LLAMA_TYPE!
)
