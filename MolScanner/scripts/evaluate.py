"""
Unified Evaluation for MolScribe_re
====================================

Evaluate any training stage on all benchmarks with configurable checkpoint.

Usage:
    python scripts/evaluate.py --stage stage1_1M680K
    python scripts/evaluate.py --stage stage1_1M680K --checkpoint epoch_10
    python scripts/evaluate.py --stage stage1_synthetic
    python scripts/evaluate.py --stage stage2_82K_MolParser --reward_mode visual
    python scripts/evaluate.py --stage stage2_82K_MolParser --reward_mode tanimoto
    python scripts/evaluate.py --checkpoint_path /path/to/model.pth --stage stage1_1M680K
    python scripts/evaluate.py --stage stage1_1M680K --gpu 0,1
"""
import sys
import json
import os
import argparse
from pathlib import Path

# Add MolScanner to Python path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir / "MolScanner"))

from MolScribe_re_model import MolScannerVocab, MolScribeModel, evaluate_benchmarks

# ===== Stage configurations =====
STAGES = {
    'stage1_1M680K': {
        'model_dir': 'MolScribe_re_1M680K',
        'description': 'Stage 1: 1M PubChem + 680K USPTO joint training',
    },
    'stage1_synthetic': {
        'model_dir': 'MolScribe_re_1M_synthetic',
        'description': 'Stage 1 (synthetic only): 1M PubChem synthetic',
    },
    'stage2_82K_MolParser': {
        'model_dir': 'MolScribe_re_82K_MolParser',
        'description': 'Stage 2: 82K MolParser RL-based fine-tuning',
    }
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate MolScribe_re on benchmarks')
    parser.add_argument('--stage', type=str, required=True,
                        choices=list(STAGES.keys()),
                        help='Training stage to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best',
                        help='Checkpoint name without .pth (default: best)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Override: explicit path to checkpoint file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--reward_mode', type=str, default='visual',
                        choices=['visual', 'tanimoto', 'edit_distance'],
                        help='Reward mode for Stage 2 checkpoint selection (default: visual)')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='Beam size for inference (default: 1)')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    data_dir = project_dir / "data"
    stage_cfg = STAGES[args.stage]

    # Stage 2 checkpoints are saved under a reward_mode subdirectory
    model_dir = Path(stage_cfg['model_dir'])
    if args.stage == 'stage2_82K_MolParser':
        model_dir = model_dir / args.reward_mode

    # ===== Resolve checkpoint path =====
    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
    else:
        ckpt_path = str(
            project_dir / "MolScanner" / "models" / model_dir / f"{args.checkpoint}.pth"
        )

    print("=" * 60)
    print(f"Evaluating: {stage_cfg['description']}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"GPU: {args.gpu}")
    print("=" * 60)

    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ===== Load model =====
    vocab = MolScannerVocab(n_bins=64)
    model = MolScribeModel(vocab=vocab, backbone='swin_b', pretrained=True)
    model.load_model(ckpt_path, device=device)

    # ===== Benchmarks =====
    benchmarks = [
        {'name': 'indigo',   'benchmark_dir': str(data_dir / "benchmark/synthetic/indigo"),   'csv_path': str(data_dir / "benchmark/synthetic/indigo.csv")},
        {'name': 'chemdraw', 'benchmark_dir': str(data_dir / "benchmark/synthetic/chemdraw"), 'csv_path': str(data_dir / "benchmark/synthetic/chemdraw.csv")},
        {'name': 'CLEF',     'benchmark_dir': str(data_dir / "benchmark/real/CLEF"),          'csv_path': str(data_dir / "benchmark/real/CLEF.csv")},
        {'name': 'JPO',      'benchmark_dir': str(data_dir / "benchmark/real/JPO"),           'csv_path': str(data_dir / "benchmark/real/JPO.csv")},
        {'name': 'coloredBG', 'benchmark_dir': str(data_dir / "benchmark/real/coloredBG"), 'csv_path': str(data_dir / "benchmark/real/coloredBG.csv")},
        {'name': 'UOB',      'benchmark_dir': str(data_dir / "benchmark/real/UOB"),           'csv_path': str(data_dir / "benchmark/real/UOB.csv")},
        {'name': 'USPTO',    'benchmark_dir': str(data_dir / "benchmark/real/USPTO"),         'csv_path': str(data_dir / "benchmark/real/USPTO.csv")},
        {'name': 'USPTO-10K',  'benchmark_dir': str(data_dir / "benchmark/real/USPTO-10K"),  'csv_path': str(data_dir / "benchmark/real/USPTO-10K.csv")},
        {'name': 'staker',   'benchmark_dir': str(data_dir / "benchmark/real/staker"),        'csv_path': str(data_dir / "benchmark/real/staker.csv")},
        {'name': 'acs',      'benchmark_dir': str(data_dir / "benchmark/real/acs"),           'csv_path': str(data_dir / "benchmark/real/acs.csv")},
        {'name': 'WildMol-10K', 'benchmark_dir': str(data_dir / "benchmark/real/WildMol-10K"), 'csv_path': str(data_dir / "benchmark/real/WildMol-10K.csv")},

    ]

    results = evaluate_benchmarks(model, benchmarks, beam_size=args.beam_size)

    # Convert any DataFrame values to list-of-dicts for JSON serialisation
    for bname, bstats in results.items():
        for k, v in bstats.items():
            if hasattr(v, 'to_dict'):
                bstats[k] = v.to_dict(orient='records')

    # ===== Save results =====
    results_dir = project_dir / "MolScanner" / "results" / model_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"{args.checkpoint}_evaluation_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # ===== Print summary =====
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for bname, bstats in results.items():
        em = bstats.get('postprocess/exact_match_acc',
             bstats.get('graph/exact_match_acc',
             bstats.get('decoder/exact_match_acc', 'N/A')))
        if isinstance(em, float):
            print(f"  {bname:12s}: {em:.4f}")
        else:
            print(f"  {bname:12s}: {em}")


if __name__ == '__main__':
    main()
