# MolScribe_re Training & Evaluation Scripts

Two-stage training pipeline for molecular structure recognition.

## Pipeline Overview

```
Stage 1: 1M PubChem + 680K USPTO joint training (supervised)  →  models/MolScribe_re_1M680K/
Stage 2: 82K MolParser RL fine-tuning (real images, no GT)     →  models/MolScribe_re_82K_MolParser/
```

Stage 1 also supports synthetic-only or USPTO-only training via `--data_mode`,
but **joint** mode is the default and recommended setting (matches original MolScribe).

## Directory Layout

```
MolScanner/
├── scripts/
│   ├── prepare_uspto_mol.py            # Download & prepare USPTO data
│   ├── train_stage1.py                 # Stage 1 unified training launcher
│   ├── train_stage2_82K_MolParser.py   # Stage 2 RL fine-tuning launcher
│   └── evaluate.py                     # Unified evaluation
├── models/
│   ├── MolScribe_re_1M680K/            # Stage 1 joint (default) checkpoints
│   │   ├── best.pth
│   │   ├── epoch_*.pth
│   │   ├── checkpoint_resume.pth
│   │   └── logs/
│   ├── MolScribe_re_1M_synthetic/      # Stage 1 synthetic-only checkpoints
│   ├── MolScribe_re_680K_USPTO/        # Stage 1 USPTO-only checkpoints
│   └── MolScribe_re_82K_MolParser/     # Stage 2 RL checkpoints
│       └── <reward_mode>/              # e.g. visual/, tanimoto/, edit_distance/
│           ├── best.pth
│           ├── epoch_*.pth
│           ├── checkpoint_resume.pth
│           └── logs/
└── results/
    ├── MolScribe_re_1M680K/            # Stage 1 joint evaluation results
    ├── MolScribe_re_1M_synthetic/      # Stage 1 synthetic-only results
    ├── MolScribe_re_680K_USPTO/        # Stage 1 USPTO-only results
    └── MolScribe_re_82K_MolParser/     # Stage 2 RL results
```

## Usage

### 1. Data Preparation

```bash
# PubChem SMILES (already prepared in data/pubchem_smiles/pubchem_smile_list.pkl)

# Download and extract USPTO Mol data (images + CSV with coordinates)
python MolScanner/scripts/prepare_uspto_mol.py
```

### 2. Stage 1: Supervised Pre-training

Joint training on 1M PubChem synthetic + 680K USPTO patent images in one
ConcatDataset — matching the original MolScribe `AuxTrainDataset` approach.

```bash
# Joint training (default, recommended)
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1.py

# Resume if interrupted
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1.py --resume

# Custom synthetic count
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1.py --synthetic_num 500000

# Synthetic-only mode
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1.py --data_mode synthetic

# USPTO-only mode (not recommended — causes catastrophic forgetting on synthetic benchmarks)
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage1.py --data_mode uspto
```

### 3. Stage 2: RL Fine-tune on 82K MolParser Real Images

Pure cycle-consistency RL fine-tuning on real-world molecule images.
No ground-truth SMILES needed — reward is computed by rendering the predicted
SMILES and comparing visual similarity with the original image.

Continues from the Stage 1 joint checkpoint by default.

```bash
# Default: REINFORCE with visual reward
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_82K_MolParser.py

# Resume if interrupted
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_82K_MolParser.py --resume

# Use a specific pretrained checkpoint
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_82K_MolParser.py \
    --pretrained_checkpoint MolScanner/models/MolScribe_re_1M680K/epoch_25.pth

# MRT method with Tanimoto reward
torchrun --nproc_per_node=4 MolScanner/scripts/train_stage2_82K_MolParser.py \
    --rl_method mrt --reward_mode tanimoto

# Quick smoke test
python MolScanner/scripts/train_stage2_82K_MolParser.py --fast_test
```

### 4. Evaluate

Evaluates on 7 benchmarks: indigo, chemdraw (synthetic) and CLEF, UOB, USPTO, staker, acs (real).

```bash
# Evaluate Stage 1 (joint, default)
python MolScanner/scripts/evaluate.py --stage stage1_1M680K

# Evaluate Stage 1 (synthetic-only variant)
python MolScanner/scripts/evaluate.py --stage stage1_synthetic

# Evaluate Stage 1 (USPTO-only variant)
python MolScanner/scripts/evaluate.py --stage stage1_uspto

# Evaluate Stage 2 (visual reward mode, default)
python MolScanner/scripts/evaluate.py --stage stage2_82K_MolParser

# Evaluate Stage 2 with a specific reward mode
python MolScanner/scripts/evaluate.py --stage stage2_82K_MolParser --reward_mode tanimoto
python MolScanner/scripts/evaluate.py --stage stage2_82K_MolParser --reward_mode edit_distance

# Evaluate a specific checkpoint
python MolScanner/scripts/evaluate.py --stage stage1_1M680K --checkpoint epoch_15

# Custom checkpoint path
python MolScanner/scripts/evaluate.py --stage stage2_82K_MolParser \
    --checkpoint_path MolScanner/models/MolScribe_re_82K_MolParser/visual/best.pth

# Beam search decoding
python MolScanner/scripts/evaluate.py --stage stage1_1M680K --beam_size 5

# Multi-GPU evaluation
python MolScanner/scripts/evaluate.py --stage stage1_1M680K --gpu 0,1
```

## Notes

- **Batch size**: Stage 1: 64 per GPU × 4 GPUs = 256 effective; Stage 2: 32 per GPU
- **Stage 1 joint data**: 1M PubChem SMILES (rendered on-the-fly by Indigo) +
  680K USPTO patent images (atom coordinates from MOL files, edges from CSV),
  combined via `ConcatDataset` — matching the original MolScribe approach
- **Stage 2 RL**: REINFORCE or MRT; reward = `w_v·𝟙[valid] + w_sim·sim(render(pred), orig) + w_e·𝟙[exact]`
  with configurable reward modes (`visual`, `tanimoto`, `edit_distance`).
  Checkpoints are saved under a subdirectory named by the reward mode
- **Stage 2 data**: MolParser sft_real (~82K real-world images) for training;
  USPTO held-out for validation
