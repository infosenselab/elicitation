# Elicitation

Elicitation is a Python package for training and evaluating Information Elicitation Agents (IEAs), based on the paper `YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents`, presented at ACL 2026.

IEAs are conversational agents designed to actively elicit information from users, rather than passively respond to requests.

## Installation

``` bash
pip install elicitation
```

## Features

-   Utilities for:
    -   Generating elicitor utterances
    -   Converting utterances into structured dialogues
-   Evaluation metrics tailored to elicitation behavior:
    -   Conformity (distributional alignment with human elicitors)
    -   Progression (forward movement of dialogue)
    -   Turn-Length Ratio (TLR) (elicitor vs. respondent verbosity)

## Examples

Example scripts and sample data are available in the `examples/` directory.

### Configuration

Create a configuration file at `config/config.yaml`:

``` yaml
paths:
  proj_store: "./proj_store"
  models: "/data/models"
```

-   `proj_store`: directory for intermediate outputs and logs
-   `models`: path to local or downloaded model checkpoints

### Running Examples

-   `example_dialoguegen.py`: End-to-end pipeline to generate elicitor utterances and convert outputs into dialogue format.
-   `example_conformity_cossim.py`: Computes Conformity (embedding similarity).
-   `example_conformity_perplexity.py`: Computes Conformity (perplexity-based).
-   `example_interaction_metrics.ipynb`: Interactive notebook for *Progression* and *Turn-Length Ratio*.

## Typical Workflow

The package follows a three-stage pipeline consistent with the YIELD evaluation setup :

1. Generate Elicitor Utterances

Use a base or fine-tuned model to generate the **next elicitor turn** given dialogue context.

``` python
from elicitation.utils import generate_utterances

output_file = generate_utterances(
    model_choice=base_model_path,
    finetuning_dataset=data_input_folder,
    model_type="finetuned",  # or "prompted"
    adapter_model=adapter_model_path,  # required if finetuned
    prompt_file=None,  # required if prompted
    batch_size=8,
    save_dir=proj_store,
)
```

2. Convert to Dialogue Format

Convert generated outputs into structured dialogues for evaluation.

``` python
from elicitation.utils import utterances_to_dialogue

utterances_to_dialogue(output_file)
```



3. Evaluate

**Conformity (Embedding Similarity)**

``` python
from elicitation.metrics import conformity_cossim

results = conformity_cossim(input_filepath, embedding_model_path)
```

**Conformity (Perplexity)**

``` python
from elicitation.metrics import conformity_perplexity

results = conformity_perplexity(
    input_filepath,
    base_model_path,
    adapter_model_path,
    use_adapter=True
)
```

**Interaction Metrics (Dialogue-Level)**

``` python
from elicitation.metrics import progression, turn_length_ratio

progression_df = progression(dialogues, embedding_model, k=2, gamma=0.9)
tlr_df = turn_length_ratio(dialogues)
```
