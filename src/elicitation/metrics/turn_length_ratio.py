from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np


def turn_length_ratio(dialogues, tokenizer_model, group_by=None, sort_by=None, export_raw=False):

    # aggregate counts across all dialogues per domain
    turn_stats = defaultdict(lambda: {"elicitor_tokens": 0, "elicitor_turns": 0,
                                    "respondent_tokens": 0, "respondent_turns": 0})

    raw_data = []
    
    for dialogue in tqdm(dialogues, desc="Turn Length"):
        
        group_value = "All" if group_by is None else dialogue.get(group_by, "Unknown")
        turns = dialogue.get("turns", [])
        
        elicitor_tokens, elicitor_turns = 0, 0
        respondent_tokens, respondent_turns = 0, 0
        
        # instead of averaging here, accumulate raw counts
        for turn in turns:
            
            role = turn.get("role", "").lower()
            utt = turn.get("utterance", "")
            
            n = count_tokens(utt, tokenizer_model)
            
            if role == "elicitor":
                elicitor_tokens += n
                elicitor_turns += 1
            
            elif role == "respondent":
                respondent_tokens += n
                respondent_turns += 1

        # accumulate per-group totals
        turn_stats[group_value]["elicitor_tokens"] += elicitor_tokens
        turn_stats[group_value]["elicitor_turns"] += elicitor_turns
        turn_stats[group_value]["respondent_tokens"] += respondent_tokens
        turn_stats[group_value]["respondent_turns"] += respondent_turns
        
        if export_raw:
            
            elicitor_avg = elicitor_tokens / elicitor_turns if elicitor_turns else 0
            respondent_avg = respondent_tokens / respondent_turns if respondent_turns else 0
            ratio = respondent_avg / elicitor_avg if elicitor_avg else np.nan

            raw_data.append({
                group_by if group_by else "All": group_value,
                "dialogue_id": dialogue.get("dialogue_id", dialogue.get("id", None)),
                "elicitor_avg_tokens": round(elicitor_avg, 2),
                "respondent_avg_tokens": round(respondent_avg, 2),
                "turn_length_ratio": round(ratio, 3),
                "elicitor_turns": elicitor_turns,
                "respondent_turns": respondent_turns
            })
        
    
    # compute per-group averages and ratios
    results = []
    
    for g, s in turn_stats.items():
        elicitor_avg = s["elicitor_tokens"] / s["elicitor_turns"] if s["elicitor_turns"] else 0
        respondent_avg = s["respondent_tokens"] / s["respondent_turns"] if s["respondent_turns"] else 0
        ratio = respondent_avg / elicitor_avg if elicitor_avg else np.nan
        results.append({
            (group_by or "All"): g,
            "elicitor_avg_tokens": round(elicitor_avg, 2),
            "respondent_avg_tokens": round(respondent_avg, 2),
            "turn_length_ratio": round(ratio, 3)
        })

    tlr_df = pd.DataFrame(results)

    if sort_by:
        tlr_df = tlr_df.sort_values(sort_by).reset_index(drop=True)
        
    if export_raw:
        raw_df = pd.DataFrame(raw_data)
        if sort_by and sort_by in raw_df.columns:
            raw_df = raw_df.sort_values(sort_by).reset_index(drop=True)
        return raw_df
    else:
        return tlr_df


#def turn_length_ratio_score(turns, tokenizer_model):
#    
#    elicitor_tokens, elicitor_turns = 0, 0
#    respondent_tokens, respondent_turns = 0, 0
#
#    for turn in turns:
#        role = turn.get("role", "").lower()
#        utt = turn.get("utterance", "")
#        
#        n = count_tokens(utt, tokenizer_model)
#        
#        if role == "elicitor":
#            elicitor_tokens += n
#            elicitor_turns += 1
#        
#        elif role == "respondent":
#            respondent_tokens += n
#            respondent_turns += 1
#
#    elicitor_avg = elicitor_tokens/elicitor_turns if elicitor_turns else 0
#    respondent_avg = respondent_tokens/respondent_turns if respondent_turns else 0
#    
#    return elicitor_avg, respondent_avg



def count_tokens(text, tokenizer):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True
        )
    )
