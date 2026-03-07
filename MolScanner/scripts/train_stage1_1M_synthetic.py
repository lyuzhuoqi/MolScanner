"""
Stage 1: Train on 1M PubChem Synthetic Data
============================================

Trains MolScribe_re from scratch on 1M dynamically generated molecular images
from PubChem SMILES using the Indigo-based drawing engine.

This is the first stage of the two-stage training pipeline:
  Stage 1: 1M PubChem synthetic  (this script)
  Stage 2: 1M PubChem + 680K USPTO  (train_stage2_1M680K_USPTO.py)

Usage:
    torchrun --nproc_per_node=4 scripts/train_stage1_1M_synthetic.py
    torchrun --nproc_per_node=4 scripts/train_stage1_1M_synthetic.py --resume
"""
import sys
from pathlib import Path
import pickle
import argparse

# Add MolScanner to Python path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir / "MolScanner"))

from MolScribe_re_model import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    args = parser.parse_args()

    data_dir = project_dir / "data"
    log_dir = project_dir / "MolScanner" / "models" / "MolScribe_re_1M_synthetic"

    # ===== Load PubChem SMILES =====
    with open(data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl", 'rb') as f:
        smiles_list = pickle.load(f)

    # ===== Determine resume checkpoint path =====
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(log_dir / "checkpoint_resume.pth")

    # ===== Fast Test Mode =====
    FAST_TEST = False

    train(
        # data
        smiles_list=smiles_list,
        smiles_num=4096 if FAST_TEST else int(1e6),

        # training
        save_path=str(log_dir),
        num_epochs=2 if FAST_TEST else 30,
        batch_size=64,
        encoder_lr=4e-4,
        decoder_lr=4e-4,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        early_stopping_patience=30,
        num_workers=4 if FAST_TEST else 8,
        seed=2026,
        use_amp=True,
        force_cpu=False,

        # molecular
        mol_augment=True,
        max_atoms=100,

        # validation (real-world benchmark)
        benchmark_dir=str(data_dir / "benchmark" / "real" / "USPTO"),
        benchmark_csv_path=str(data_dir / "benchmark" / "real" / "USPTO.csv"),
        val_max_samples=50 if FAST_TEST else None,

        # vision transformer
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        pretrained=True,

        # decoder transformer
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # resume
        resume_from=resume_path,
    )
