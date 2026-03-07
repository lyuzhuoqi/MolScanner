"""
DDP Training Launcher for MolSctibe_re
=====================================
Usage:
    torchrun --nproc_per_node=4 train_MolScribe_re.py
    torchrun --nproc_per_node=4 train_MolScribe_re.py --resume

Key differences from DataParallel version:
  - batch_size is PER-GPU (effective = batch_size * 4 GPUs = 256)
  - Each GPU runs its own process (no GIL contention)
  - Gradient synchronization via NCCL all-reduce (overlapped with backward)
  - ~30-50% faster than DataParallel
"""
from MolScribe_re_model import train
from pathlib import Path
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    log_dir = project_dir / "MolScanner" / "models" / "MolScribe_re_1M_synthetic"
    data_dir = project_dir / "data"
    file_path = data_dir / "pubchem_smiles/pubchem_smile_list.pkl"
    with open(file_path, 'rb') as f:
        smiles_list = pickle.load(f)
    
    smiles_all = smiles_list

    # ===== Determine resume checkpoint path =====
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(log_dir / "checkpoint_resume.pth")

    # ===== Fast Test Mode =====
    FAST_TEST = False  # Set to False for full training
    
    train(
        # sample
        smiles_list=smiles_list,
        smiles_num=4096 if FAST_TEST else int(1e6),
        
        # training
        save_path=str(log_dir),
        num_epochs=2 if FAST_TEST else 30,
        # NOTE: batch_size is PER-GPU now!
        # With 4 GPUs: effective_batch = 64 * 4 = 256
        batch_size=64,
        encoder_lr=4e-4,
        decoder_lr=4e-4,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        early_stopping_patience=30,
        # NOTE: num_workers is PER-PROCESS (each GPU has its own DataLoader)
        # 8 workers * 4 processes = 32 total
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
        dim_feedforward=4*256,
        dropout=0.1,

        # resume
        resume_from=resume_path,
    )
