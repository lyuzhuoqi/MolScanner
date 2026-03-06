"""
Real-Image RL Finetuning Launcher
==================================

Pure cycle-consistency RL finetuning on real-world molecule images.
No MLE loss — only REINFORCE with visual reward:
    pred SMILES → render → encode → cosine_sim(features_orig, features_rendered)

Trains on real benchmark images (staker, UOB, CLEF, acs),
validates on USPTO (held out).

Usage:
    # Multi-GPU (4x L40)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_rl_real.py

    # Single-GPU fast test
    python train_rl_real.py --fast_test

    # Resume
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_rl_real.py --resume
"""
from MolScribe_re_model import train_rl_real_finetune
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-image RL finetuning')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to specific checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke-test with tiny data')
    parser.add_argument('--rl_method', type=str, default='reinforce',
                        choices=['reinforce', 'mrt'],
                        help='RL method: reinforce or mrt')
    parser.add_argument('--mrt_alpha', type=float, default=1.0,
                        help='MRT alpha (sharpening exponent)')
    parser.add_argument('--reward_mode', type=str, default='visual',
                        choices=['visual', 'tanimoto', 'edit_distance'],
                        help='Reward similarity mode')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    molscanner_dir = project_dir / "MolScanner"
    real_dir = data_dir / "benchmark" / "real"

    # Save directory
    save_dir = molscanner_dir / "models" / "MolScribe_re_real_rl"

    # Pretrained model
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        pretrained_path = str(molscanner_dir / "models" / "MolScribe_re"
                              / "best.pth")

    # Training data: staker (50K) + UOB (5.7K) + CLEF (992) + acs (331)
    # Validation: USPTO (5.7K) — held out
    train_csv_paths = [
        str(real_dir / "staker.csv"),
        str(real_dir / "UOB.csv"),
        str(real_dir / "CLEF.csv"),
        str(real_dir / "acs.csv"),
    ]
    train_image_dirs = [
        str(real_dir / "staker"),
        str(real_dir / "UOB"),
        str(real_dir / "CLEF"),
        str(real_dir / "acs"),
    ]

    # Resume checkpoint
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(save_dir / "checkpoint_resume.pth")

    FAST_TEST = args.fast_test

    train_rl_real_finetune(
        # data
        train_csv_paths=train_csv_paths,
        train_image_dirs=train_image_dirs,

        # validation (USPTO held out)
        val_benchmark_dir=str(real_dir / "USPTO"),
        val_benchmark_csv=str(real_dir / "USPTO.csv"),
        val_max_samples=50 if FAST_TEST else None,

        # model
        pretrained_path=pretrained_path,
        save_path=str(save_dir),

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
        num_epochs=2 if FAST_TEST else 20,
        batch_size=8 if FAST_TEST else 16,
        encoder_lr=1e-5,
        decoder_lr=1e-5,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        seed=2026,
        early_stopping_patience=10,
        num_workers=2 if FAST_TEST else 4,
        use_amp=True,
        force_cpu=False,

        # RL
        rl_max_len=500,
        rl_temperature=0.8,
        rl_n_samples=16,
        rl_subsample=16,

        # Reward weights:
        # R = w_v·𝟙[valid] + w_sim·cosine_sim(render(pred), orig) + w_e·𝟙[exact]
        reward_validity_weight=0.1,
        reward_similarity_weight=0.5,
        reward_exact_match_weight=0.4,
        reward_mode=args.reward_mode,

        # RL method
        rl_method=args.rl_method,
        mrt_alpha=args.mrt_alpha,

        # resume
        resume_from=resume_path,
    )
