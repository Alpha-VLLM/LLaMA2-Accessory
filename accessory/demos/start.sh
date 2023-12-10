#!/bin/bash
PYTHONPATH="../:$PYTHONPATH"
export PYTHONPATH

# ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Optional: Interactive model downloading
while true; do
  echo -e "${YELLOW}Do you want to download the model? (y/n):${NC}"
  read DOWNLOAD
  case $DOWNLOAD in
    [Yy]* )
      echo -e "${GREEN}Downloading the model...${NC}"
      python tools/download.py
      break
      ;;
    [Nn]* ) break;;
    * ) echo -e "${RED}Please answer y or n.${NC}";;
  esac
done

# Choose the inference scenario
while true; do
  echo -e "${YELLOW}Choose the inference scenario:${NC}"
  echo -e "${GREEN}1) Single-turn Dialogue (Single-modal)${NC}"
  echo -e "${GREEN}2) Multi-turn Dialogue (Single-modal)${NC}"
  echo -e "${GREEN}3) Single-turn Dialogue (Multi-modal)${NC}"
  echo -e "${GREEN}4) Multi-turn Dialogue (Multi-modal)${NC}"  
  echo -e "${GREEN}5) Multi-turn Dialogue (Multi-modal) Box Mode${NC}"
  echo -e "${BLUE}6) Single-model Command Line Interface (CLI)${NC}"
  echo -e "${BLUE}7) Multi-model Command Line Interface (CLI)${NC}"
  read -p "Please enter the option number: " SCENARIO
  case $SCENARIO in
    1|2)
      OPTIONS=("internlm" "llama_adapter" "llama_peft" "llama")
      break
      ;;
    3|4|5)  # Updated case for multi-modal options
      OPTIONS=("llama_qformerv2" "llama_qformerv2_peft" "llama_ens" "llama_ens5" "llama_ens10" "llama_ens5p2" "llama_ens_peft")
      break
      ;;
    6)
  echo -e "${BLUE}Running Single-model CLI...${NC}"
  python demos/single_model_cli.py
  ;;
    7)
  echo -e "${BLUE}Running Multi-model CLI...${NC}"
  python demos/multi_model_cli.py
  ;;
    * ) echo -e "${RED}Invalid option. Please re-enter.${NC}";;
  esac
done

# Choose llama_type by index with check
while true; do
  echo -e "${YELLOW}Choose llama_type:${NC}"
  for i in "${!OPTIONS[@]}"; do 
    echo -e "${GREEN}$((i+1))) ${OPTIONS[$i]}${NC}"
  done
  read -p "Please enter the option number: " CHOICE
  if [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le ${#OPTIONS[@]} ]; then
    LLAMA_TYPE="${OPTIONS[$((CHOICE-1))]}"
    break
  else
    echo -e "${RED}Invalid choice. Please re-enter.${NC}"
  fi
done

# Ask if the user wants to enable Quantization with check
while true; do
  echo -e "${YELLOW}Do you want to enable Quantization? (y/n):${NC}"
  read QUANT
  case $QUANT in
    [Yy]* ) QUANT="--quant"; break;;
    [Nn]* ) QUANT=""; break;;
    * ) echo -e "${RED}Please answer y or n.${NC}";;
  esac
done

# Ask for the model size to determine the number of processors per node
while true; do
  echo -e "${YELLOW}Enter the option number for model size (1: 7B, 2: 13B, 3: 34B, 4: 70B):${NC}"
  read SIZE_CHOICE
  case $SIZE_CHOICE in
    1) SIZE="7B"; NPROC=1; break;;
    2) SIZE="13B"; NPROC=2; break;;
    3) SIZE="34B"; NPROC=4; break;;
    4) SIZE="70B"; NPROC=8; break;;
    * ) echo -e "${RED}Invalid choice. Please re-enter.${NC}";;
  esac
done

# Check the master port is within 0-65536
while true; do
  echo -e "${YELLOW}Enter the master port (default is 12345):${NC}"
  read PORT
  if [ -z "$PORT" ]; then
    PORT=12345
    break
  elif [[ "$PORT" -ge 0 ]] && [[ "$PORT" -le 65536 ]]; then
    break
  else
    echo -e "${RED}Invalid port number(0~65536). Please re-enter.${NC}"
  fi
done

# Check if PARAMS, TOKENIZER, PRETRAINED exist
while true; do
  echo -e "${YELLOW}Enter the path to params.json:${NC}"
  read PARAMS
  if [ -f "$PARAMS" ]; then break; else echo -e "${RED}File not found. Please re-enter.${NC}"; fi
done

while true; do
  echo -e "${YELLOW}Enter the path to tokenizer.model:${NC}"
  read TOKENIZER
  if [ -f "$TOKENIZER" ]; then break; else echo -e "${RED}File not found. Please re-enter.${NC}"; fi
done

while true; do
  echo -e "${YELLOW}Enter the path to the pretrained model:${NC}"
  read PRETRAINED
  if [ -d "$PRETRAINED" ]; then break; else echo -e "${RED}Directory not found. Please re-enter.${NC}"; fi
done

# Special handling for peft models
if [[ $LLAMA_TYPE == *"peft"* ]]; then
  PARAMS="$PARAMS configs/model/finetune/sg/llamaPeft_normBiasLora.json"
fi

# Run the corresponding Python script based on user's choice
case $SCENARIO in
  1)
    echo -e "${GREEN}Running single_turn.py...${NC}"
    torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/single_turn.py --llama_config $PARAMS --tokenizer_path $TOKENIZER --pretrained_path $PRETRAINED $QUANT --llama_type $LLAMA_TYPE
    ;;
  2)
    echo -e "${GREEN}Running multi_turn.py...${NC}"
    python  demos/multi_turn.py --n_gpus=$NPROC --master_port=$PORT --llama_config $PARAMS --tokenizer_path $TOKENIZER --pretrained_path $PRETRAINED $QUANT --llama_type $LLAMA_TYPE
    ;;
  3)
    echo -e "${GREEN}Running single_turn_mm.py...${NC}"
    torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/single_turn_mm.py --llama_config $PARAMS --tokenizer_path $TOKENIZER --pretrained_path $PRETRAINED $QUANT --llama_type $LLAMA_TYPE
    ;;
  4)  # New case for Multi-turn Dialogue (Multi-modal)
    echo -e "${GREEN}Running multi_turn_mm.py...${NC}"
    torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/multi_turn_mm.py --llama_config $PARAMS --tokenizer_path $TOKENIZER --pretrained_path $PRETRAINED $QUANT --llama_type $LLAMA_TYPE
    ;;
  5)  # New case for Multi-turn Dialogue (Multi-modal) Box Mode
    echo -e "${GREEN}Running multi_turn_mm_box.py...${NC}"
    torchrun --nproc-per-node=$NPROC --master-port=$PORT demos/multi_turn_mm_box.py --llama_config $PARAMS --tokenizer_path $TOKENIZER --pretrained_path $PRETRAINED $QUANT --llama_type $LLAMA_TYPE
    ;;
  *)
    echo -e "${RED}Invalid option. Exiting.${NC}"
    ;;
esac
