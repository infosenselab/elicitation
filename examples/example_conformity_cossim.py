# ---------------- Imports ----------------
import os
from datetime import datetime

import yaml

from elicitation.metrics import conformity_cossim


# ---------------- Arguments ----------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
INPUT_FILE = "20251013T1603-llama-3.2-3b-instruct-finetuned"


# ---------------- Config ----------------
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


proj_store = config["paths"]["proj_store"]
models_folderpath = config["paths"]["models"]

input_filepath = os.path.join(proj_store, "evaluation", "generated-utterances", f"{INPUT_FILE}.jsonl")
embedding_model_filepath = os.path.join(models_folderpath, EMBEDDING_MODEL)


input_folder = os.path.dirname(input_filepath)
parent_folder = os.path.dirname(input_folder)
output_path = os.path.join(parent_folder, "conformity", "cossim")
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, f"{INPUT_FILE}.csv")



# ---------------- Main ----------------
def main():
        
    conformity_cossim_results = conformity_cossim(input_filepath, embedding_model_filepath)
    print(conformity_cossim_results)        
    
    conformity_cossim_results = conformity_cossim(input_filepath, embedding_model_filepath, group_by="domain", sort_by="domain", export_raw=True)
    print(conformity_cossim_results)
    
    conformity_cossim_results = conformity_cossim(input_filepath, embedding_model_filepath, group_by="domain", sort_by="domain")
    print(conformity_cossim_results)
    
    conformity_cossim_results.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    
    main()



