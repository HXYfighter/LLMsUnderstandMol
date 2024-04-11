# LLMsUnderstandMol
This is the code repository for our paper [Empirical Evidence for the Fragment level understanding on Drug Molecular Structure of LLMs](https://arxiv.org/abs/2401.07657), which was accepted by [AAAI 2024 LLMs4Bio workshop](https://llms4science-community.github.io/aaai2024.html).

## Dependencies

```bash
pytorch==1.12.1
rdkit==2020.03
tqdm
tensorboard
guacamol
```

The ChEMBL dataset is available at [ChEMBL](https://www.ebi.ac.uk/chembl/).

## Pre-training

```bash
python codes/pretrain.py 
```

## Reinforcement Learning

```bash
python codes/generator_guacamol.py --task_id 0
python codes/generator_guacamol.py --task_id 1
python codes/generator_guacamol.py --task_id 2
```

## Analysis of Fragments

See the `.ipynb` notebooks.
