# ---------------- Imports ----------------
import os
import json
import glob
import logging

from datetime import datetime

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm


# ---------------- Functions ----------------
def format_prompted_conversation(messages, custom_system_prompt):
    # First message should be system
    system_message = ""
    if messages and messages[0]["role"] == "system":
        system_message = messages[0]["content"].strip()
        messages = messages[1:]

    transcript_lines = []
    for m in messages:
        if m["role"] == "user":
            transcript_lines.append(f"respondent: {m['content'].strip()}")
        elif m["role"] == "assistant":
            transcript_lines.append(f"elicitor: {m['content'].strip()}")
        else:
            raise ValueError(f"Unexpected role {m['role']}")

    full_text = system_message
    if custom_system_prompt:
        full_text += "\n" + custom_system_prompt
    full_text += "\n\n" + "\n".join(transcript_lines)

    return full_text


def initialize_pipeline(base_model_path, model_type, adapter_model_path=None):

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto"
    )

    if model_type == "finetuned":
        model = PeftModel.from_pretrained(base_model, adapter_model_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500), tokenizer





# ---------------- Main ----------------
def obtain_utterances(pipe, tokenizer, jsonl_files, model_type, prompt_file_text, output_file, batch_size, logger):


    results = []


    # Collect all input_texts to process
    input_prompts = []
    meta_data = []  # to store associated block_ids etc

    total_lines = 0
    for file in jsonl_files:
        with open(file, 'r', encoding='utf-8') as fin:
            total_lines += sum(1 for _ in fin)

    progress_bar = tqdm(total=total_lines, desc="Preparing prompts")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                obj = json.loads(line)
                domain = obj.get("domain", "")
                messages = obj.get("messages", [])

                # Ensure last message is assistant, otherwise error out
                if not messages or messages[-1]["role"] != "assistant":
                    raise ValueError(f"Block {obj.get('block_id')} does not end with an assistant message.")

                # Context = everything up to final assistant
                context_messages = messages[:-1]
                
                # Real response = last message
                real_response = messages[-1]["content"].strip()

                if model_type == "prompted":
                    #input_text = format_prompted_conversation(context_messages, prompt_file_text)
                    context_messages[0]["content"] += "\n\n" + prompt_file_text 
                #else:
                
                # Build input text with chat template
                input_text = tokenizer.apply_chat_template(
                    context_messages,
                    tokenize=False,
                    add_generation_prompt=True # Final assistant already removed
                )
                    
                
                # Appends
                input_prompts.append(input_text)
                meta_data.append({
                    "block_id": obj.get("block_id"),
                    "domain": domain,
                    "context_messages": context_messages,
                    "real_response": real_response,
                    
                })


                progress_bar.update(1)

    progress_bar.close()

    # Inference in batches
    inference_bar = tqdm(total=len(input_prompts), desc="Running inference")

    for i in range(0, len(input_prompts), batch_size):
        batch_inputs = input_prompts[i:i+batch_size]
        outputs = pipe(batch_inputs)

        for j, output in enumerate(outputs):
            generated_raw = output[0]['generated_text']

            # Remove the prompt prefix if present
            if generated_raw.startswith(batch_inputs[j]):
                generated_response = generated_raw[len(batch_inputs[j]):].strip()
            else:
                generated_response = generated_raw.strip()
            meta = meta_data[i + j]
            results.append({
                "block_id": meta["block_id"],
                "domain": meta["domain"],
                "context_messages": meta["context_messages"],
                "real_response": meta["real_response"],
                "generated_response": generated_response
            })
            
            # Logging
            logger.info(f"Input Formatted Text: {batch_inputs[j]}")
            logger.info(f"Real Response: {meta['real_response']}")
            logger.info(f"Generated Response: {generated_response}")
            

        inference_bar.update(len(batch_inputs))

    inference_bar.close()

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in results:
            json.dump(item, fout)
            fout.write('\n')

    print(f"\nFinished! Saved {len(results)} generations to {output_file}")
    
    return output_file




# ---------------- Execution ----------------

def generate_utterances(
    model_choice,
    finetuning_dataset,
    model_type,
    adapter_model=None,
    prompt_file=None,
    batch_size=8,
    save_dir = "./"
):
    
    
    # ---------------- Validation ----------------
    if model_type not in ["finetuned", "prompted"]:
        raise ValueError("model_type must be 'finetuned' or 'prompted'")

    if model_type == "finetuned" and not adapter_model:
        raise ValueError("adapter_model must be specified for finetuned mode")

    if model_type == "prompted" and not prompt_file:
        raise ValueError("prompt_file must be specified for prompted mode")

    if model_type == "prompted" and adapter_model is not None:
        raise ValueError("adapter_model must be None for prompted mode")

    prompt_file_text = ""
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_file_text = f.read().strip()

    
    
    # ---------------- Config ----------------
    timestamp = datetime.now().strftime("%Y%m%dT%H%M")
    #output_filename = os.path.join(save_dir, "evaluation", "generated-utterances", f"{timestamp}-{model_choice.split('/')[-1].lower().replace('_', '-')}-{model_type}.jsonl")
    
    output_filepath = os.path.join(save_dir, "evaluation", "generated-utterances")
    if model_type == "prompted":
        output_filename = os.path.join(output_filepath, f"{timestamp}-{model_choice.split('/')[-1].lower().replace('_', '-')}-pr.jsonl")
    elif model_type == "finetuned":
        output_filename = os.path.join(output_filepath, f"{timestamp}-{adapter_model.split('/')[-1].lower().replace('_', '-')}.jsonl")
    else:
        raise ValueError("model_type must be either 'prompted' or 'finetuned'")
    
    

    

    
    # ---------------- Logging ----------------
    timestamp = datetime.now().strftime("%Y%m%dT%H%M")
    logs_dir = os.path.join(save_dir, "logs", "generate-utterances")
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f"{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename)]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Using model: {model_choice}")
    logger.info(f"Dataset: {finetuning_dataset}")
    logger.info(f"Adapter model: {adapter_model}")
    logger.info(f"Mode: {model_type}")
    
    
    
    # Input JSONL files
    jsonl_files = glob.glob(os.path.join(finetuning_dataset, "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {finetuning_dataset}")

    
    # ---------------- Run ----------------
    pipe, tokenizer = initialize_pipeline(model_choice, model_type, adapter_model)
    output_filename = obtain_utterances(pipe, tokenizer, jsonl_files, model_type, prompt_file_text, output_filename, batch_size, logger)

    return output_filename




