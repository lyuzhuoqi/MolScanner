from MolScanner.MolScanner_model_archive import train
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
    
    train(
        # sample
        smiles_list=smiles_list,
        smiles_num=4096 if FAST_TEST else int(1e6),
        
        # training
        save_path=str(log_dir),
        num_epochs=2 if FAST_TEST else 30,
        batch_size=256,
        learning_rate=4e-4,
        warmup_ratio=0.02,
        early_stopping_patience=30,
        num_workers=4 if FAST_TEST else 16,
        seed=2026,
        use_amp=True,
        force_cpu=False,
        # bond_chunk_size=16,
        
        # molecular
        mol_augment=True,
        max_atoms=100,
        train_atom_shuffle=False,

        # validation (real-world benchmark)
        benchmark_dir=str(data_dir / "benchmark" / "real" / "USPTO"),
        benchmark_csv_path=str(data_dir / "benchmark" / "real" / "USPTO.csv"),
        val_max_samples=50 if FAST_TEST else None,  # None = use all samples

        # vision transformer
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        pretrained=True,

        # decoder transformer
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=256,
        dropout=0.1,
    )