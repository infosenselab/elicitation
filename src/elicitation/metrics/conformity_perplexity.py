# ---------------- Imports ----------------
import json
import math
import sys
import os

from datetime import datetime

import yaml
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# Match your generation dtype and mapping
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#CONTEXT_KEYS = ["input_prompt", "model_input", "prompt", "context", "input_text"]


# ------------------------
# HELPERS
# ------------------------

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_context(obj, tokenizer):

    messages = obj.get("context_messages", [])
    if not messages or not isinstance(messages, list):
        raise KeyError(f"Missing or invalid 'context_messages' for block_id={obj.get('block_id')}")

    # Use the chat template to generate properly formatted model input
    context = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # adds the final assistant stub
    )
    return context




def _truncate_context_to_fit(ctx_ids, tgt_ids, max_len):
    """
    Keep the full target. If needed, drop tokens from the left of the context.
    """
    ctx_len = ctx_ids.shape[1]
    tgt_len = tgt_ids.shape[1]
    total = ctx_len + tgt_len
    if total <= max_len:
        return ctx_ids, tgt_ids

    remove = total - max_len
    if remove >= ctx_len:
        # Keep at least one target token if extreme truncation happens
        keep_tgt = max(1, max_len)
        tgt_ids = tgt_ids[:, -keep_tgt:]
        ctx_ids = ctx_ids[:, :0]
    else:
        ctx_ids = ctx_ids[:, remove:]
    return ctx_ids, tgt_ids

@torch.no_grad()
def nll_loss_for_target(tokenizer, model, context_text, target_text):
    """Compute negative log-likelihood loss and perplexity for target tokens only."""
    ctx = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)
    tgt = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)

    max_len = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    if max_len is None or max_len == float("inf"):
        max_len = 4096

    # Truncate from left if over max length
    ctx_len = ctx["input_ids"].shape[1]
    tgt_len = tgt["input_ids"].shape[1]
    total_len = ctx_len + tgt_len
    if total_len > max_len:
        remove = total_len - max_len
        ctx["input_ids"] = ctx["input_ids"][:, remove:]
        ctx["attention_mask"] = ctx["attention_mask"][:, remove:]

    # Concatenate context + target
    input_ids = torch.cat([ctx["input_ids"], tgt["input_ids"]], dim=1).to(model.device)
    attention_mask = torch.cat([ctx["attention_mask"], tgt["attention_mask"]], dim=1).to(model.device)

    # Mask out context tokens in the loss
    labels = input_ids.clone()
    labels[:, :ctx["input_ids"].shape[1]] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss.item()
    ppl = math.exp(loss)
    target_tokens = (labels != -100).sum().item()

    return loss, ppl, target_tokens


def conformity_perplexity(input_filepath, base_model_path, adapter_model_path=None, use_adapter=False,
                          group_by=None, sort_by=None, export_raw=False):
    """Compute perplexity on real elicitor responses conditioned on dialogue context."""
    print(f"\nLoading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_adapter and adapter_model_path:
        print(f"Loading adapter weights from: {adapter_model_path}")
        model = PeftModel.from_pretrained(model, adapter_model_path)
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

    model.eval()

    results = []
    with open(input_filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scoring perplexity"):
            obj = json.loads(line)
            block_id = obj.get("block_id")
            domain = obj.get("domain", "unknown").strip()
            messages = obj.get("context_messages", [])
            real_response = obj.get("real_response", "").strip()

            if not real_response or not messages:
                continue

            # Proper chat-format for context
            context_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            try:
                loss, ppl, n_tokens = nll_loss_for_target(tokenizer, model, context_text, real_response)
                results.append({
                    "block_id": block_id,
                    "domain": domain,
                    "loss": loss,
                    "perplexity": ppl,
                    "target_token_count": n_tokens
                })
            except Exception as e:
                results.append({
                    "block_id": block_id,
                    "domain": domain,
                    "loss": float("nan"),
                    "perplexity": float("nan"),
                    "target_token_count": 0,
                    "error": str(e)
                })

    df = pd.DataFrame(results)

    #  Aggregation 
    if export_raw:
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)
        return df

    group_col = "all" if group_by is None else group_by
    grouped = df.groupby(group_col) if group_by else [("all", df)]

    summary_rows = []
    for name, group in grouped:
        valid = group.dropna(subset=["loss", "perplexity"])
        if len(valid) == 0:
            continue

        denom = valid["target_token_count"].sum()
        micro_loss = (valid["loss"] * valid["target_token_count"]).sum() / denom if denom > 0 else float("nan")
        micro_ppl = math.exp(micro_loss) if not math.isnan(micro_loss) else float("nan")

        summary_rows.append({
            group_col: name,
            "n": len(valid),
            "micro_loss": round(micro_loss, 4),
            "micro_perplexity": round(micro_ppl, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    if sort_by and sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(sort_by).reset_index(drop=True)

    return summary_df
