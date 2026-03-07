# MolScribe_re Training & Evaluation Scripts

Two-stage training pipeline following the original MolScribe approach.

## Pipeline Overview

```
Stage 1: 1M PubChem Synthetic (dynamic rendering)       →  models/MolScribe_re_1M_synthetic/
Stage 2: 680K USPTO Mol (patent images + mol-file coords) →  models/MolScribe_re_680K_USPTO/
```

## Directory Layout

```
MolScanner/
├── scripts/
│   ├── prepare_uspto_mol.py            # Download & prepare USPTO data
│   ├── train_stage1_1M_synthetic.py    # Stage 1 training launcher
│   ├── train_stage2_680K_USPTO.py      # Stage 2 training launcher
│   └── evaluate.py                     # Unified evaluation
├── models/
│   ├── MolScribe_re_1M_synthetic/      # Stage 1 checkpoints & logs
│   │   ├── best.pth
│   │   ├── epoch_*.pth
│   │   ├── checkpoint_resume.pth
│   │   └── logs/
│   └── MolScribe_re_680K_USPTO/        # Stage 2 checkpoints & logs
│       ├── best.pth
│       ├── epoch_*.pth
│       ├── checkpoint_resume.pth
│       └── logs/
└── results/
    ├── MolScribe_re_1M_synthetic/      # Stage 1 evaluation results
    │   └── best_evaluation_results.json
    └── MolScribe_re_680K_USPTO/        # Stage 2 evaluation results
        └── best_evaluation_results.json
```

## Usage

### 1. Data Preparation

```bash
# PubChem SMILES (already prepared in data/pubchem_smiles/pubchem_smile_list.pkl)

# Download and extract USPTO Mol data (images + CSV with coordinates)
python MolScanner/scripts/prepare_uspto_mol.py
```

### 2. Stage 1: Train on 1M Synthetic (already done)

```bash
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1_1M_synthetic.py

# Resume if interrupted
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1_1M_synthetic.py --resume
```

### 3. Stage 2: Fine-tune on 680K USPTO (patent images)

```bash
# First run: loads Stage 1 best.pth, fresh optimizer
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_680K_USPTO.py

# Resume if interrupted
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_680K_USPTO.py --resume

# Use a specific Stage 1 checkpoint
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_680K_USPTO.py \
    --stage1_checkpoint MolScanner/models/MolScribe_re_1M_synthetic/epoch_30.pth
```

### 4. Evaluate

```bash
# Evaluate Stage 1
python MolScanner/scripts/evaluate.py --stage stage1_1M_synthetic

# Evaluate Stage 2
python MolScanner/scripts/evaluate.py --stage stage2_680K_USPTO

# Evaluate a specific checkpoint
python MolScanner/scripts/evaluate.py --stage stage2_680K_USPTO --checkpoint epoch_15

# Multi-GPU evaluation
python MolScanner/scripts/evaluate.py --stage stage2_680K_USPTO --gpu 0,1
```

## Notes

- **Batch size**: 64 per GPU × 4 GPUs = 256 effective (matches original)
- **Stage 1 data**: SMILES rendered on-the-fly by the Indigo drawing engine
- **Stage 2 data**: Original patent images from USPTO; atom coordinates extracted
  from MOL files; edge/bond labels from the CSV. Loaded via `USPTOMolDataset`
- **Stage 2 fine-tuning**: Loads Stage 1 model weights with a fresh optimizer/scheduler,
  following a cosine annealing schedule from scratch
