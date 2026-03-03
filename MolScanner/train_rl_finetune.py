"""
RL Finetuning Launcher for MolScribe_re
=======================================

Finetunes a pretrained MolScribe model with a combined loss:
    L_total = w_t·L_token + w_b·L_bond + α(t) · L_REINFORCE_Tanimoto

Usage:
    # Multi-GPU (4x L40)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_rl_finetune.py

    # Resume
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_rl_finetune.py --resume

    # Single-GPU fast test
    python train_rl_finetune.py --fast_test

Key design choices:
  - RL loss computed every N MLE steps (saves ~80% sampling cost)
  - α linearly annealed from 0 → α_max over warmup epochs
  - Running-average baseline for variance reduction
  - Lower LR than pretraining (1e-5 vs 4e-4) to avoid catastrophic forgetting
  - rl_subsample controls how many images per batch are used for RL (memory cap)
"""
from MolScribe_re_model import train_rl_finetune
from pathlib import Path
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL finetuning for MolScribe')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from rl_finetune/checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to specific checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights (.pth)')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke-test with tiny data')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    molscanner_dir = project_dir / "MolScanner"

    # Save directory for RL finetuning outputs
    save_dir = molscanner_dir / "models" / "MolScribe_re_rl"

    # Pretrained model
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        pretrained_path = str(molscanner_dir / "models" / "MolScribe_re" / "20260226_edge_padding_finished.pth")

    # Training data
    file_path = data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl"
    with open(file_path, 'rb') as f:
        smiles_list = pickle.load(f)

    # Resume checkpoint
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(save_dir / "checkpoint_resume.pth")

    FAST_TEST = args.fast_test

    train_rl_finetune(
        # data
        smiles_list=smiles_list,
        smiles_num=2048 if FAST_TEST else int(5e5),

        # training
        save_path=str(save_dir),
        pretrained_path=pretrained_path,
        num_epochs=2 if FAST_TEST else 10,
        # Per-GPU batch size: effective = 32 * 4 GPUs = 128
        batch_size=16 if FAST_TEST else 32,
        encoder_lr=1e-5,
        decoder_lr=1e-5,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        seed=2026,
        early_stopping_patience=10,
        num_workers=2 if FAST_TEST else 8,
        use_amp=True,
        force_cpu=False,

        # molecular
        mol_augment=True,
        max_atoms=100,

        # validation
        benchmark_dir=str(data_dir / "benchmark" / "real" / "USPTO"),
        benchmark_csv_path=str(data_dir / "benchmark" / "real" / "USPTO.csv"),
        val_max_samples=30 if FAST_TEST else None,

        # architecture (must match pretrained model)
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # ===== Loss weights =====
        token_loss_weight=1.0,   # weight for token CE loss
        bond_loss_weight=1.0,    # weight for bond CE loss

        # ===== RL hyperparameters =====
        alpha_rl_max=0.5,          # max RL weight after warmup
        alpha_rl_warmup_epochs=0, # linearly anneal alpha from 0 → max over N epochs
        rl_every_n_steps=5,      # compute RL loss every N MLE steps (cost control)
        rl_max_len=500,          # max decode length for RL sampling (match pretraining)
        rl_temperature=0.8,      # sampling temperature (lower = less noisy)
        rl_n_samples=4,          # samples per image (set >1 for self-critical baseline)
        rl_subsample=32,         # max images per batch for RL sampling (memory cap)

        # resume
        resume_from=resume_path,
    )
