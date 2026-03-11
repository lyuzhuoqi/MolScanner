"""
Stage 1: Supervised Pre-training on Synthetic + USPTO Data
==========================================================

Trains MolScribe_re from scratch.  Three data modes, matching the original
MolScribe training strategy:

  --data_mode joint      (default)  1M PubChem synthetic + 680K USPTO patent images
  --data_mode synthetic             1M PubChem synthetic only
  --data_mode uspto                 680K USPTO patent images only (not recommended)

The "joint" mode combines dynamically rendered PubChem SMILES with real USPTO
patent images in a single ConcatDataset, following the original MolScribe
AuxTrainDataset approach and avoiding catastrophic forgetting.

Prerequisites (for joint / USPTO modes):
    USPTO data prepared: python scripts/prepare_uspto_mol.py

Usage:
    # Joint training (recommended, matches original MolScribe)
    torchrun --nproc_per_node=4 scripts/train_stage1.py

    # Synthetic only
    torchrun --nproc_per_node=4 scripts/train_stage1.py --data_mode synthetic

    # USPTO only (not recommended)
    torchrun --nproc_per_node=4 scripts/train_stage1.py --data_mode uspto

    # Resume if interrupted
    torchrun --nproc_per_node=4 scripts/train_stage1.py --resume

    # Custom synthetic count
    torchrun --nproc_per_node=4 scripts/train_stage1.py --synthetic_num 500000
"""
import sys
import os
from pathlib import Path
import pickle
import argparse

# Add MolScanner to Python path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir / "MolScanner"))

from MolScribe_re_model import train

# Model directory names per data mode
MODEL_DIRS = {
    'joint':     'MolScribe_re_1M680K',
    'synthetic': 'MolScribe_re_1M_synthetic',
    'uspto':     'MolScribe_re_680K_USPTO',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 1: Supervised pre-training on synthetic and/or USPTO data')
    parser.add_argument('--data_mode', type=str, default='joint',
                        choices=['joint', 'synthetic', 'uspto'],
                        help='Data mode: joint (1M+680K, default), synthetic (1M), or uspto (680K)')
    parser.add_argument('--synthetic_num', type=int, default=int(1e6),
                        help='Number of PubChem SMILES to use (default: 1M)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--val_benchmarks', type=str, nargs='+',
                        default=['USPTO', 'JPO'],
                        help='Validation benchmark names (default: USPTO JPO). '
                             'First one is primary for early stopping.')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_dir = project_dir / "data"
    log_dir = project_dir / "MolScanner" / "models" / MODEL_DIRS[args.data_mode]

    # ===== Load PubChem SMILES (for joint / synthetic modes) =====
    smiles_list = None
    if args.data_mode in ('joint', 'synthetic'):
        smiles_path = data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl"
        if not smiles_path.exists():
            print(f"ERROR: PubChem SMILES not found: {smiles_path}")
            sys.exit(1)
        with open(smiles_path, 'rb') as f:
            smiles_list = pickle.load(f)
        smiles_num = min(args.synthetic_num, len(smiles_list))
        print(f"PubChem SMILES pool: {len(smiles_list)}, using {smiles_num}")

    # ===== Locate USPTO CSV (for joint / uspto modes) =====
    train_csv = None
    if args.data_mode in ('joint', 'uspto'):
        uspto_dir = data_dir / "uspto_mol"
        csv_candidates = [
            uspto_dir / "train_680k_normalised.csv",
            uspto_dir / "train_680k.csv",
            uspto_dir / "train.csv",
        ]
        for c in csv_candidates:
            if c.exists():
                train_csv = str(c)
                break
        if train_csv is None and uspto_dir.exists():
            csvs = sorted(uspto_dir.glob("*.csv"))
            if csvs:
                train_csv = str(csvs[0])
        if train_csv is None:
            print(f"ERROR: No USPTO training CSV found in {uspto_dir}")
            print("Run first: python scripts/prepare_uspto_mol.py")
            sys.exit(1)
        print(f"USPTO training CSV: {train_csv}")

    # ===== Determine resume checkpoint path =====
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(log_dir / "checkpoint_resume.pth")

    # ===== Fast Test Mode =====
    FAST_TEST = False

    # ===== Build train() kwargs based on data mode =====
    data_kwargs = {}
    if args.data_mode == 'synthetic':
        # Pure PubChem synthetic: dynamic rendering from SMILES
        data_kwargs = dict(
            pubchem_smiles_list=smiles_list,
            pubchem_smiles_num=4096 if FAST_TEST else smiles_num,
        )
    elif args.data_mode == 'uspto':
        # Pure USPTO: file-based patent images (not recommended)
        data_kwargs = dict(
            uspto_csv_path=train_csv,
            uspto_data_dir=str(data_dir),
        )
    else:
        # Joint: USPTO images + PubChem synthetic via ConcatDataset
        data_kwargs = dict(
            pubchem_smiles_list=smiles_list,
            pubchem_smiles_num=4096 if FAST_TEST else smiles_num,
            uspto_csv_path=train_csv,
            uspto_data_dir=str(data_dir),
        )

    # ===== Build validation benchmark list =====
    real_dir = data_dir / "benchmark" / "real"
    val_benchmarks = []
    for bm_name in args.val_benchmarks:
        bm_dir = real_dir / bm_name
        bm_csv = real_dir / f"{bm_name}.csv"
        if not bm_csv.exists():
            print(f"WARNING: benchmark CSV not found: {bm_csv}, skipping")
            continue
        val_benchmarks.append({'name': bm_name, 'dir': str(bm_dir), 'csv': str(bm_csv)})

    print(f"Data mode: {args.data_mode}")
    print(f"Save path: {log_dir}")

    train(
        **data_kwargs,

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

        # validation (real-world benchmarks)
        benchmarks=val_benchmarks,
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
