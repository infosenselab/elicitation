# ---------------- Imports ----------------
import json
import os



def utterances_to_dialogue(input_file, ):
    
    
    # Prep
    file_name = os.path.basename(input_file) 
    input_folder = os.path.dirname(input_file)
    parent_folder = os.path.dirname(input_folder)

    output_folderpath = os.path.join(parent_folder, "generated-utterances-dialogue")
    os.makedirs(output_folderpath, exist_ok=True)

    # Define output file paths
    real_output_file = os.path.join(
        output_folderpath, file_name.replace(".jsonl", ""), "real", "dialogue.json"
    )
    gen_output_file = os.path.join(
        output_folderpath, file_name.replace(".jsonl", ""), "generated", "dialogue.json"
    )

    os.makedirs(os.path.dirname(real_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(gen_output_file), exist_ok=True)

    
    # Convert    
    real_dialogues = []
    gen_dialogues = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            dialogue_id = data["block_id"]
            domain = data["domain"]
            context_messages = data["context_messages"]

            # Build dialogue turns but skip system messages
            turns = []
            turn_id = 0
            for msg in context_messages:
                if msg["role"] == "system":
                    continue  # discard system messages

                speaker = "assistant" if msg["role"] == "assistant" else "user"
                role = "elicitor" if msg["role"] == "assistant" else "respondent"

                turns.append({
                    "turn_id": turn_id,
                    "timestamp": "",
                    "speaker": speaker,
                    "role": role,
                    "utterance": msg["content"]
                })
                turn_id += 1

            # Append real response as last assistant turn
            real_turns = turns + [{
                "turn_id": turn_id,
                "timestamp": "",
                "speaker": "assistant",
                "role": "elicitor",
                "utterance": data["real_response"]
            }]

            # Append generated response as last assistant turn
            gen_turns = turns + [{
                "turn_id": turn_id,
                "timestamp": "",
                "speaker": "assistant",
                "role": "elicitor",
                "utterance": data["generated_response"]
            }]

            # Add to dialogue lists
            real_dialogues.append({
                "dialogue_id": dialogue_id,
                "domain": domain,
                "turns": real_turns
            })
            gen_dialogues.append({
                "dialogue_id": dialogue_id,
                "domain": domain,
                "turns": gen_turns
            })

    # Write outputs
    with open(real_output_file, "w", encoding="utf-8") as f:
        json.dump(real_dialogues, f, indent=2, ensure_ascii=False)

    with open(gen_output_file, "w", encoding="utf-8") as f:
        json.dump(gen_dialogues, f, indent=2, ensure_ascii=False)


    print(f"\nFinished!. Saved real dialogues to {real_output_file} and generated dialogues to {gen_output_file}")
