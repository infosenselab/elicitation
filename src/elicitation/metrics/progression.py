import numpy as np
from elicitation.metrics.utils import cosine
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def progression(dialogues, embedding_model, device, k=5, gamma=0.99, group_by=None, sort_by=None, export_raw=False):

    group_shift_scores = defaultdict(list)
    raw_scores = []
    
    for dialogue in tqdm(dialogues, desc="Progression"):
        group_value = "all" if group_by is None else dialogue.get(group_by, "Unknown")
        turns = dialogue.get("turns", [])
        utterances = [t.get("utterance", "") for t in turns if t.get("utterance", "")]
        if len(utterances) < 2:
            continue
        
        embeddings = embedding_model.encode(utterances, device=device, show_progress_bar=False)
        avg_shift, shift_values = progression_score(embeddings, k=k, gamma=gamma, return_all=True)
        
        group_shift_scores[group_value].append(avg_shift)
        
        if export_raw:
            for idx, score in enumerate(shift_values):
                raw_scores.append({
                    group_by if group_by else "all": group_value,
                    "dialogue_id": dialogue.get("dialogue_id", dialogue.get("id", None)),
                    "pair_index": idx,
                    "progression_score": score
                })
        
        
    
    group_column = "all" if group_by is None else group_by

    progression_df = pd.DataFrame([
        {group_column: g, "dialogues": len(s), "progression": np.mean(s)}
        for g, s in group_shift_scores.items()
    ])

    if sort_by:
        progression_df = progression_df.sort_values(sort_by).reset_index(drop=True)

    if export_raw:
        raw_df = pd.DataFrame(raw_scores)
        if sort_by and sort_by in raw_df.columns:
            raw_df = raw_df.sort_values(sort_by).reset_index(drop=True)
        return raw_df
    else:
        return progression_df


def progression_score(segment, k, gamma, return_all=False):
    vals = []
    length = len(segment)
    for t in range(k, length):
        num, denom = 0.0, 0.0
        for j in range(1, k+1):
            if t - j < 0:
                break
            w = gamma**j
            num += w * (1 - cosine(segment[t-j], segment[t]))
            denom += w
        if denom > 0:
            vals.append(num/denom)
    
    if return_all:
        avg_val = np.mean(vals) if vals else 0.0
        return avg_val, vals
    
    return np.mean(vals) if vals else 0.0



