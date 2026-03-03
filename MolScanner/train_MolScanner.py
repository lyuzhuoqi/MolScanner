"""
DDP Training Launcher for MolScanner with Cycle-Consistency Image Loss
=======================================================================
Usage:
    torchrun --nproc_per_node=4 train_MolScanner_cycle_consistency.py

This launcher extends the base MolScanner training with an optional
cycle-consistency image loss:
  1. The model greedy-decodes a subset of each batch → predicted graphs
  2. Predicted graphs are rendered back into molecule images (Indigo)
  3. A frozen pretrained MolDiscriminator compares originals vs reconstructed
  4. The discriminator embedding distance is used as a REINFORCE reward

Key parameters:
  - discriminator_path: path to a pretrained MolDiscriminator checkpoint
  - image_loss_weight: λ in total = supervised + λ·image_loss
  - image_loss_every_n_steps: how often to compute the image loss
"""
from MolScanner_model import train
from pathlib import Path
import pickle

if __name__ == '__main__':
    project_dir = Path(__file__).parent.parent
    log_dir = project_dir / "MolScanner" / "models" / "MolScanner"
    data_dir = project_dir / "data"
    file_path = data_dir / "pubchem_smiles/pubchem_smile_list.pkl"
    with open(file_path, 'rb') as f:
        smiles_list = pickle.load(f)
    
    smiles_all = smiles_list

    # ===== Fast Test Mode =====
    FAST_TEST = False  # Set to False for full training
    
    # ===== MolDiscriminator checkpoint =====
    disc_path = str(project_dir / "MolScanner" / "models" / "MolDiscriminator" / "20260223_curriculum.pth")
    
    # ===== Resume from pretrained checkpoint (set to None for training from scratch) =====
    # Example: resume a pretrained MolScribe_re model for RL fine-tuning:
    #   resume_path = str(project_dir / "MolScanner" / "models" / "MolScanner" / "best.pth")
    resume_path = str(project_dir / "MolScanner" / "models" / "MolScribe_re" / "20260226_edge_padding_finished.pth")
    
    train(
        # sample
        smiles_list=smiles_list,
        smiles_num=4096 if FAST_TEST else int(1e6),
        
        # training
        save_path=str(log_dir),
        num_epochs=2 if FAST_TEST else 10,
        # NOTE: batch_size is PER-GPU now!
        # With 4 GPUs: effective_batch = 64 * 4 = 256
        batch_size=64,
        encoder_lr=1e-4,
        decoder_lr=1e-4,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        early_stopping_patience=10,
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
        
        # ===== checkpoint resume =====
        resume_from=resume_path,
        
        # ===== cycle-consistency image loss =====
        discriminator_path=disc_path,
        image_loss_weight=1,
        image_loss_every_n_steps=5 if FAST_TEST else 20,
        image_loss_max_samples=16,
        image_loss_max_decode_len=300,
        image_loss_num_samples_per_image=5,
        image_loss_temperature=0.5,
        image_loss_start_epoch=1 if resume_path else 3,
    )
