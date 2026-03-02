from MolDiscriminator_model import MoleculeDiscriminator
from pathlib import Path
import pickle
import argparse


def _parse_optional_patience(value: str):
    value_norm = value.strip().lower()
    if value_norm in {'none', 'null'}:
        return None
    try:
        patience = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "early_stopping_patience must be an integer or 'none'"
        ) from e
    if patience < 1:
        raise argparse.ArgumentTypeError(
            "early_stopping_patience must be >= 1, or 'none'"
        )
    return patience


def parse_args():
    parser = argparse.ArgumentParser(description='Train molecule discriminator')
    parser.add_argument('--num_smiles', type=int, default=1e6,
                        help='Number of SMILES to sample (default: use 1 million)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='Learning rate (default: 4e-4)')
    parser.add_argument('--val_split', type=float, default=0.01,
                        help='Validation split fraction (default: 0.01)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers per process (default: 8)')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed (default: 2025)')
    parser.add_argument('--early_stopping_patience',
                        type=_parse_optional_patience, default=None,
                        help="Early stopping patience in epochs, or 'none' to disable (default: none)")
    parser.add_argument('--no_ddp', action='store_true',
                        help='Disable DDP even with multiple GPUs')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Triplet loss margin (default: 1.0)')
    parser.add_argument('--image_size', type=int, default=384,
                        help='Image size (square, default: 384)')
    parser.add_argument('--warmup_ratio', type=float, default=0.02,
                        help='LR warmup ratio (default: 0.02)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (automatic mixed precision)')
    parser.add_argument('--curriculum_epochs', type=int, default=None,
                        help='Number of epochs for curriculum ramp '
                             '(default: half of num_epochs)')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke test with 100 SMILES, 2 epochs, '
                             'batch_size=8, 1 worker, no DDP')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume training from this epoch (0 = start fresh). '
                             'Loads from epoch_N.pth or training_state.pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    project_dir = Path(__file__).parent.parent
    log_dir = project_dir / "MolScanner" / "models" / "MolDiscriminator"
    data_dir = project_dir / "data"
    file_path = data_dir / "pubchem_smiles/pubchem_smile_list.pkl"
    with open(file_path, 'rb') as f:
        smiles_list = pickle.load(f)

    # Override settings for fast smoke test
    if args.fast_test:
        print('=== FAST TEST MODE ===')
        args.num_smiles = 100
        args.num_epochs = 2
        args.batch_size = 8
        args.num_workers = 1
        args.no_ddp = False
        args.val_split = 0.2

    # Initialize discriminator
    discriminator = MoleculeDiscriminator(
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        image_size=(args.image_size, args.image_size),
    )

    # Train with DDP
    losses = discriminator.train(
        smiles_list=smiles_list,
        num_smiles=args.num_smiles,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mol_augment=True,
        hard_negative_max_prob=1.0,
        hard_negative_min_prob=0.5,
        curriculum_epochs=args.curriculum_epochs,
        save_path=str(log_dir),
        val_split=args.val_split,
        val_mol_augment=False,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        use_ddp=not args.no_ddp,
        warmup_ratio=args.warmup_ratio,
        use_amp=not args.no_amp,
        resume_epoch=args.resume_epoch,
    )