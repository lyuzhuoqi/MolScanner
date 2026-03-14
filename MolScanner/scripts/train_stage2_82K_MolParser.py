"""
Stage 2: Real-Image RL Finetuning on 82K MolParser Data
========================================================

Pure cycle-consistency RL finetuning on real-world molecule images.
No MLE loss — only MRT with visual reward:
    pred SMILES → render → encode → cosine_sim(features_orig, features_rendered)

Continues from the Stage 1 (1M+680K joint) checkpoint.
Trains on MolParser sft_real data (~82K), validates on USPTO (held out).

Prerequisites:
    1. Stage 1 trained: models/MolScribe_re_1M680K/best.pth exists
    2. MolParser sft_real data available: data/molparser_sft_real/

Usage:
    torchrun --nproc_per_node=4 scripts/train_stage2_82K_MolParser.py
    torchrun --nproc_per_node=4 scripts/train_stage2_82K_MolParser.py --resume
    python scripts/train_stage2_82K_MolParser.py --fast_test
"""
import sys
import os
from pathlib import Path
import argparse

# Add MolScanner to Python path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir / "MolScanner"))

from MolScribe_re_model import train_rl_finetune

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-image RL finetuning')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pretrained model weights for fine-tuning '
                             '(default: models/MolScribe_re_1M680K/best.pth)')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke-test with tiny data')
    parser.add_argument('--mrt_alpha', type=float, default=1.0,
                        help='MRT alpha (sharpening exponent)')
    parser.add_argument('--reward_mode', type=str, default='visual',
                        choices=['visual', 'tanimoto', 'edit_distance'],
                        help='Reward similarity mode')
    parser.add_argument('--visual_reward_renderer', type=str, default='rdkit',
                        choices=['rdkit', 'drawing_engine'],
                        help='Renderer backend for visual reward images')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--val_benchmarks', type=str, nargs='+',
                        default=['USPTO', 'JPO'],
                        help='Validation benchmark names (default: USPTO JPO). '
                             'First one is primary for early stopping.')
    parser.add_argument('--visual_reward_encoder', type=str, default=None,
                        help='Path to external image encoder checkpoint (e.g. '
                             'MolDiscriminator SiameseNetwork .pth) for visual reward')
    parser.add_argument('--visual_reward_embedding_dim', type=int, default=128,
                        help='Embedding dim of the external visual reward encoder (default: 128)')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_dir = project_dir / "data"
    log_dir = project_dir / "MolScanner" / "models" / "MolScribe_re_82K_MolParser" / args.reward_mode
    real_dir = data_dir / "benchmark" / "real"

    # ===== Determine checkpoint paths =====
    pretrained_path = args.pretrained_checkpoint or str(
        project_dir / "MolScanner" / "models" / "MolScribe_re_1M680K" / "best.pth"
    )

    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(log_dir / "checkpoint_resume.pth")

    # ===== Build validation benchmark list =====
    val_benchmarks = []
    for bm_name in args.val_benchmarks:
        bm_dir = real_dir / bm_name
        bm_csv = real_dir / f"{bm_name}.csv"
        if not bm_csv.exists():
            print(f"WARNING: benchmark CSV not found: {bm_csv}, skipping")
            continue
        val_benchmarks.append({'name': bm_name, 'dir': str(bm_dir), 'csv': str(bm_csv)})

    # Training data: MolParser sft_real (~82K)
    train_csv_paths = [
        str(data_dir / "molparser_sft_real" / "sft_real.csv"),
    ]
    train_image_dirs = [
        str(data_dir / "molparser_sft_real" / "images"),
    ]

    # ===== Fast Test Mode =====
    FAST_TEST = args.fast_test

    train_rl_finetune(
        # data
        train_csv_paths=train_csv_paths,
        train_image_dirs=train_image_dirs,

        # validation benchmarks
        benchmarks=val_benchmarks,
        val_max_samples=50 if FAST_TEST else None,

        # model
        pretrained_path=pretrained_path,
        save_path=str(log_dir),

        # architecture (must match pretrained)
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # training
        num_epochs=2 if FAST_TEST else 10,
        batch_size=4 if FAST_TEST else 32,
        encoder_lr=1e-5,
        decoder_lr=1e-5,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        seed=2026,
        early_stopping_patience=10,
        num_workers=2 if FAST_TEST else 4,
        use_amp=True,

        # RL
        rl_max_len=500,
        rl_temperature=0.8,
        rl_n_samples=16,

        # Reward weights:
        # R = w_v·𝟙[valid] + w_sim·cosine_sim(render(pred), orig) + w_e·𝟙[exact]
        reward_validity_weight=0.1,
        reward_similarity_weight=0.5,
        reward_exact_match_weight=0.4,
        reward_mode=args.reward_mode,
        visual_reward_renderer=args.visual_reward_renderer,

        mrt_alpha=args.mrt_alpha,

        # External visual reward encoder
        visual_reward_encoder_path=args.visual_reward_encoder,
        visual_reward_embedding_dim=args.visual_reward_embedding_dim,

        # resume
        resume_from=resume_path,
    )
