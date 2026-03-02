from MolScribe_re_model import MolScannerVocab, MolScribeModel, evaluate_benchmarks
from pathlib import Path
import json

if __name__ == '__main__':
    proj_dir = Path.home() / "projects" / "Markush"
    data_dir = proj_dir / "data"

    vocab = MolScannerVocab(n_bins=64)
    model = MolScribeModel(vocab=vocab, backbone='swin_b', pretrained=True)
    checkpoint = 'best'
    model.load_model(str(proj_dir / "MolScanner" / "models" / "MolScribe_re" / f"{checkpoint}.pth"), device='cuda')

    benchmarks = [
        {'name': 'indigo', 'benchmark_dir': str(data_dir / "benchmark/synthetic/indigo"), 'csv_path': str(data_dir / "benchmark/synthetic/indigo.csv")},
        {'name': 'chemdraw', 'benchmark_dir': str(data_dir / "benchmark/synthetic/chemdraw"), 'csv_path': str(data_dir / "benchmark/synthetic/chemdraw.csv")},
        {'name': 'CLEF',  'benchmark_dir': str(data_dir / "benchmark/real/CLEF"),  'csv_path': str(data_dir / "benchmark/real/CLEF.csv")},
        {'name': 'UOB',   'benchmark_dir': str(data_dir / "benchmark/real/UOB"),   'csv_path': str(data_dir / "benchmark/real/UOB.csv")},
        {'name': 'USPTO', 'benchmark_dir': str(data_dir / "benchmark/real/USPTO"), 'csv_path': str(data_dir / "benchmark/real/USPTO.csv")},
        {'name': 'staker', 'benchmark_dir': str(data_dir / "benchmark/real/staker"), 'csv_path': str(data_dir / "benchmark/real/staker.csv")},
        {'name': 'acs',   'benchmark_dir': str(data_dir / "benchmark/real/acs"),   'csv_path': str(data_dir / "benchmark/real/acs.csv")},
    ]

    results = evaluate_benchmarks(model, benchmarks, beam_size=1)

    # Convert any DataFrame values to list-of-dicts for JSON serialisation
    for bname, bstats in results.items():
        for k, v in bstats.items():
            if hasattr(v, 'to_dict'):          # pd.DataFrame / pd.Series
                bstats[k] = v.to_dict(orient='records')

    output_file = proj_dir / "MolScanner" / "results" / "MolScribe_re_evaluation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)