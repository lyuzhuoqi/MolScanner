"""
Stage 2: Fine-tune on 680K USPTO Mol (patent images + mol-file coordinates)
============================================================================

Continues from the Stage 1 (1M synthetic) checkpoint.  Unlike Stage 1 which
dynamically renders images from SMILES, Stage 2 uses the *original* patent
images and atom coordinates extracted from MOL files — matching the vanilla
MolScribe training setup.

Prerequisites:
    1. Stage 1 trained: models/MolScribe_re_1M_synthetic/best.pth exists
    2. USPTO data prepared: python scripts/prepare_uspto_mol.py

Usage:
    torchrun --nproc_per_node=4 scripts/train_stage2_680K_USPTO.py
    torchrun --nproc_per_node=4 scripts/train_stage2_680K_USPTO.py --resume
"""
import sys
import os
from pathlib import Path
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
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Path to Stage 1 model weights for fine-tuning '
                             '(default: models/MolScribe_re_1M_synthetic/best.pth)')
    args = parser.parse_args()

    data_dir = project_dir / "data"
    log_dir = project_dir / "MolScanner" / "models" / "MolScribe_re_680K_USPTO"

    # ===== Locate USPTO CSV =====
    # The CSV is expected at data/uspto_mol/train_680k.csv (or train.csv)
    uspto_dir = data_dir / "uspto_mol"
    csv_candidates = [
        uspto_dir / "train_680k.csv",
        uspto_dir / "train.csv",
    ]
    train_csv = None
    for c in csv_candidates:
        if c.exists():
            train_csv = str(c)
            break
    if train_csv is None:
        # Try any CSV in the directory
        if uspto_dir.exists():
            csvs = sorted(uspto_dir.glob("*.csv"))
            if csvs:
                train_csv = str(csvs[0])
    if train_csv is None:
        print(f"ERROR: No USPTO training CSV found in {uspto_dir}")
        print("Run first: python scripts/prepare_uspto_mol.py")
        sys.exit(1)
    print(f"USPTO training CSV: {train_csv}")

    # ===== Determine checkpoint paths =====
    stage1_path = args.stage1_checkpoint or str(
        project_dir / "MolScanner" / "models" / "MolScribe_re_1M_synthetic" / "best.pth"
    )

    resume_path = None
    finetune_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(log_dir / "checkpoint_resume.pth")
    else:
        # First run: fine-tune from Stage 1 checkpoint
        finetune_path = stage1_path

    # ===== Fast Test Mode =====
    FAST_TEST = False

    train(
        # data: file-based USPTO mol data
        train_csv_path=train_csv,
        train_data_dir=str(data_dir),

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
        max_atoms=100,

        # validation (real-world benchmark)
        benchmark_dir=str(data_dir / "benchmark" / "real" / "USPTO"),
        benchmark_csv_path=str(data_dir / "benchmark" / "real" / "USPTO.csv"),
        val_max_samples=50 if FAST_TEST else None,

        # vision transformer (must match Stage 1)
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        pretrained=True,

        # decoder transformer (must match Stage 1)
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # checkpoint loading
        resume_from=resume_path,
        finetune_from=finetune_path,
    )
