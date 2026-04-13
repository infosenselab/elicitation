# ---------------- Imports ----------------
import sys
import os
import json
import glob
import logging

from datetime import datetime

import yaml
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm

from elicitation.utils import generate_utterances, utterances_to_dialogue


# ---------------- Arguments ----------------
BATCH_SIZE = 8
MODEL_TYPE = "finetuned"

# Paths
MODEL_CHOICE = "meta-llama/Llama-3.2-3B-Instruct"
FINETUNING_DATASET = "yield-v1-small1pct-finetuning"
ADAPTER_MODEL = "20251012T2228-llama-3.2-3b-instruct"
PROMPT_FILE = None #"./config/generation_prompt.txt"



# ---------------- Config ----------------
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj_store"]

data_path = os.path.join("./sample_data")
data_input_folder = os.path.join(data_path, FINETUNING_DATASET, "test") # Only on the test set

models_folderpath = config["paths"]["models"]
base_model_path = f"{models_folderpath}/{MODEL_CHOICE}"

# Saved model and tokenizer path
adapter_model_path = os.path.join(proj_store, "experiments", "fine-tuning", "adapter-models", ADAPTER_MODEL)


# ---------------- Main ----------------
def main():
    
    # Generate utterances
    output_file = generate_utterances(
        model_choice = base_model_path,
        finetuning_dataset = data_input_folder,
        model_type = MODEL_TYPE,
        adapter_model = adapter_model_path,
        prompt_file = PROMPT_FILE,
        batch_size = BATCH_SIZE,
        save_dir = proj_store,
    )

    # Convert them to dialigue form
    utterances_to_dialogue(output_file)



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    main()



