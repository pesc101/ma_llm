# On Improving Repository-Level Code QA for Large Language Models

We present this repository that includes all information regarding the paper.
The repository is split up into six folders, each providing information.

1. Dataset: Includes SpyderCodeQA, the evaluation dataset created.
2. Evaluation: All Scripts for the evaluation using LLM-as-a-Judge.
3. RAG: Everything related to the creation of the RAG pipeline
4. Self-Alignment: Includes all prompts and script for creating the Self-Alignment process.
5. Train: Everything related to fine-tuning Mistral 7B using SFT and QLoRA
6. Shared: General Classes used in several other modules.


## SpyderCodeQA

The dataset consists of three different dimensions:

1. Source Code Semantics (N = 140)
2. Dependencies (N = 135)
3. Meta-Information (N = 50)

The process of creating the dataset can be read in the paper.

## LLM-as-a-Judge Evaluation

The evaluation can be executed using:
 ```
bash evaluation/scripts/eval.sh
```
The paths of the models had to be set. Models can be saved locally are HF paths can be used.
In addition, parameter for the models and rag pipeline can be set.
The path for the dataset, the paths where the created answers should be saved as well as the paths to the system prompts should be added.

The rest is executed automatically and the results are saved to:
```
evaluation/results/answers/{timestamp}
```

## RAG Pipeline

The RAG pipeline can be initalized using the following bash script:
```
bash rag/scripts/init_collections.sh
```

Provide the input path of the dataset and everything should work.

## Self-Alignment

For executing the Self-Alignment the script to use is `self_alignment.sh`. Set all parameter in the bash script and it should work out of the box.


```bash
bash selfalign/scripts/self_alignment.sh [option] [additional_parameters]
bash selfalign/scripts/self_alignment.sh  ## for complete self-alignment
bash selfalign/scripts/self_alignment.sh augmented ## for only self-augmentation
bash selfalign/scripts/self_alignment.sh curation results/aug/2023-11-17 ## for only self-augmentation
```
