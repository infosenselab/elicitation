# ---------------- Imports ----------------
import os
import json

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer




# ---------------- Functions ----------------
def context_response_length(input_filepath, tokenizer_model, group_by=None, sort_by=None, export_raw=False):
    
    base_model_name = os.path.splitext(os.path.basename(input_filepath))[0]

    tokenizer_name = os.path.basename(tokenizer_model.rstrip("/"))
    print(f"\nLoading tokenizer: {tokenizer_name}")
    

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    all_results = []



    with open(input_filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Computing token lengths"):
            if not line.strip():
                continue

            obj = json.loads(line)

            block_id = obj.get("block_id")
            domain = obj.get("domain", "").strip()

            # ---- Context ----
            context_messages = obj.get("context_messages", [])
            
            # Proper chat-format for context
            context_text = tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            
            #context_text = "\n".join(
            #    m.get("content", "") for m in context_messages
            #)

            # ---- Responses ----
            real_response = obj.get("real_response", "").strip()
            generated_response = obj.get("generated_response", "").strip()

            if not real_response or not generated_response:
                continue

            #print(f"real_response: {real_response}")
            real_tokens = count_tokens(real_response, tokenizer)
            
            #print(f"context_text: {context_text}")
            context_tokens = count_tokens(context_text, tokenizer)
            
            #print(f"generated_response: {generated_response}")
            generated_tokens = count_tokens(generated_response, tokenizer)
            
            

            group_value = "all" if group_by is None else obj.get(group_by, "Unknown")

            all_results.append({
                "block_id": block_id,
                group_by if group_by else "group": group_value,
                "domain": domain,
                "base_model": base_model_name,
                "tokenizer": tokenizer_name,
                "context_tokens": context_tokens,
                "real_response_tokens": real_tokens,
                "generated_response_tokens": generated_tokens,
                "real_response": real_response,
                "generated_response": generated_response
            })

    df = pd.DataFrame(all_results)

    # ---- Raw output ----
    if export_raw:
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)
        return df

    # ---- Aggregate ----
    group_column = group_by if group_by else "group"

    summary_df = (
        df.groupby(group_column, as_index=False)
        .agg(
            n=("block_id", "count"),
            avg_context_tokens=("context_tokens", "mean"),
            avg_real_response_tokens=("real_response_tokens", "mean"),
            avg_generated_response_tokens=("generated_response_tokens", "mean"),
        )
    )

    if group_by is None:
        summary_df = summary_df.rename(columns={"group": "all"})

    if sort_by and sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(sort_by).reset_index(drop=True)

    return summary_df



def count_tokens(text, tokenizer):
    encoded = tokenizer.encode(
            text,
            add_special_tokens=False # already added with tokenizer.apply_chat_template
        )
    
    # AUDIT
    #decoded = tokenizer.decode(
    #    encoded,
    #    skip_special_tokens=False
    #)
    #print(decoded)
    
    return len(encoded)