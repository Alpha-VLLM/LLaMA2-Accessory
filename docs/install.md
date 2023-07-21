# Environment Setup
* Setup up a new conda env and install required packages
  ```bash
  # create conda env
  conda create -n accessory python=3.10 -y
  conda activate accessory
  # install packages
  pip install -r requirements.txt
  ```
* This project relies on [apex](https://github.com/NVIDIA/apex), which needs to be compiled from source. Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source).
