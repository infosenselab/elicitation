# ---------------- Imports ----------------
from elicitation.metrics.utils import cosine

import os
import json

import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ---------------- Functions ----------------
def conformity_cossim(input_filepath, embedding_model_filepath, group_by=None, sort_by=None, export_raw=False):

    base_model_name = os.path.splitext(os.path.basename(input_filepath))[0]
    
    # Derive embedding model name from the model object (not from the global constant)
    embedding_model_name = os.path.basename(embedding_model_filepath.rstrip("/"))
    print(f"\nLoading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_filepath)
    
    
    all_results = []

    with open(input_filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Computing similarities"):
            if not line.strip():
                continue

            obj = json.loads(line)
            block_id = obj.get("block_id")
            domain = obj.get("domain", "").strip()
            real_response = obj.get("real_response", "").strip()
            generated_response = obj.get("generated_response", "").strip()

            if not real_response or not generated_response:
                continue

            sim_score = compute_similarity(real_response, generated_response, embedding_model)
            
            # Determine grouping value
            group_value = "all" if group_by is None else obj.get(group_by, "Unknown")

            all_results.append({
                "block_id": block_id,
                group_by if group_by else "group": group_value,
                "domain": domain,
                "base_model": base_model_name,
                "embedding_model": embedding_model_name,
                "similarity": sim_score,
                "real_response": real_response,
                "generated_response": generated_response
            })

    df = pd.DataFrame(all_results)

    # Default (export_raw=True): return the full per-sample table
    if export_raw:
        # Optionally sort if requested
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)
        return df

    # If export_raw=False: aggregate into a summary table
    group_column = group_by if group_by else "group"

    summary_df = (
        df.groupby(group_column, as_index=False)
        .agg(
            n=("similarity", "count"),
            conformity_cossim=("similarity", "mean"),
        )
    )
    
    # Rename "group" column to "all" when no group_by was provided
    if group_by is None:
        summary_df = summary_df.rename(columns={"group": "all"})

    # Apply sorting if requested
    if sort_by and sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(sort_by).reset_index(drop=True)

    return summary_df



def compute_similarity(text1, text2, embedding_model):
    emb1 = embedding_model.encode(text1)
    emb2 = embedding_model.encode(text2)
    return 1 - cosine(emb1, emb2)



