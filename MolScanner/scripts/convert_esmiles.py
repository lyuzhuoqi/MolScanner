"""
Convert MolParser E-SMILES to wild SMILES-like strings for RL fine-tuning.

E-SMILES format:
    backbone_SMILES<sep><a>atom_idx:label</a>...<r>atom_idx:rgroup</r>...

Design rationale:
    • Keep substituents / abbreviations / R-group labels as displayed labels
        instead of collapsing to bare '*'.
    • Do NOT canonicalize here. Canonicalization/legality checks can be handled
        later by reward-specific logic (e.g., Tanimoto path).
    • Output may be illegal for RDKit but can be rendered by Indigo-based
        wild-SMILES drawing.

Conversion rules:
  DROP entries whose tag section contains <r>, </r>, <c>, </c>, or ?
      — these encode structural relationships not representable in SMILES.
    ACCEPT all <a> labels (maximally inclusive):
        1. Pure SMILES (empty after <sep>): use backbone directly (no canonicalization).
        2. <dum> (attachment point): convert to [R].
        3. Any other label on a '*' atom: keep verbatim as [LABEL].
  Subscript brackets (R[1]→R1, CH[2]→CH2) and primes (R'→R) are normalised.
    Final output is NOT canonicalized and NOT sanitized.

Usage:
    cd /path/to/Markush && python convert_esmiles.py
"""
import os
import sys
import re
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
 
_SUBSCRIPT_RE = re.compile(r'\[(\w+)\]')   # R[1] → R1, CH[2] → CH2


def _normalize_label(label: str) -> str:
    """Strip E-SMILES subscript brackets and trailing primes."""
    label = _SUBSCRIPT_RE.sub(r'\1', label)
    label = label.rstrip("'")
    return label


def _replace_star_labels(backbone: str, a_tags):
    """Replace labeled '*' atoms in backbone with [LABEL], keeping order stable.

    The replacement is done by assigning temporary isotopes to target '*' atoms,
    exporting non-canonical SMILES, then replacing [n*] -> [LABEL].
    """
    mol = Chem.MolFromSmiles(backbone, sanitize=False)
    if mol is None:
        return None

    mol_rw = Chem.RWMol(mol)

    # atom_idx -> normalized label
    star_labels = {}
    for idx_str, raw_label in a_tags:
        idx = int(idx_str)
        if idx >= mol_rw.GetNumAtoms():
            return None
        atom = mol_rw.GetAtomWithIdx(idx)
        if atom.GetSymbol() != '*':
            continue
        label = _normalize_label(raw_label)
        if label in ('<dum>', 'dum', ''):
            label = 'R'  # attachment point → [R]
        star_labels[idx] = label

    if not star_labels:
        return backbone

    next_iso = 1001
    token_to_label = {}
    for idx, label in star_labels.items():
        atom = mol_rw.GetAtomWithIdx(idx)
        atom.SetIsotope(next_iso)
        token_to_label[f'[{next_iso}*]'] = f'[{label}]'
        next_iso += 1

    out = Chem.MolToSmiles(mol_rw, canonical=False)
    # Replace longer tokens first for safety.
    for token in sorted(token_to_label.keys(), key=len, reverse=True):
        out = out.replace(token, token_to_label[token])
    return out


def esmiles_to_smiles(esmiles: str):
    """Convert a single E-SMILES string to wild SMILES-like output.

    Returns (smiles | None, category) where category is one of:
    'pure', 'wild', 'drop_tag', 'fail'
    """
    if '<sep>' not in esmiles:
        return None, 'drop_tag'

    backbone, tags_str = esmiles.split('<sep>', 1)
    tags_str = tags_str.strip()

    # 1. Pure SMILES — empty after <sep>
    if not tags_str:
        return backbone, 'pure'

    # 2. Drop structural annotations not representable in SMILES
    if any(tok in tags_str for tok in ('<r>', '</r>', '<c>', '</c>', '?')):
        return None, 'drop_tag'

    # 3. Parse <a> tags
    a_tags = re.findall(r'<a>(\d+):(.+?)</a>', tags_str)
    if not a_tags:
        return backbone, 'pure'

    out = _replace_star_labels(backbone, a_tags)
    if out is None:
        return None, 'fail'
    return out, 'wild'


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'molparser_sft_real')
    output_csv = os.path.join(data_dir, 'sft_real.csv')

    print("Loading MolParser-7M sft_real from HuggingFace cache...")
    from datasets import load_dataset
    ds = load_dataset("UniParser/MolParser-7M", name="sft_real", split="train")
    print(f"Total entries: {len(ds)}")

    converted = []
    counts = {'pure': 0, 'wild': 0, 'drop_tag': 0, 'fail': 0}

    for i, item in enumerate(tqdm(ds, desc="Converting E-SMILES")):
        esmiles = item['SMILES']
        smiles, category = esmiles_to_smiles(esmiles)
        counts[category] += 1

        if smiles is not None:
            converted.append({
                'image_id': str(i),
                'file_path': f'molparser_sft_real/images/{i}.png',
                'SMILES': smiles,
            })

    df_out = pd.DataFrame(converted)
    df_out.to_csv(output_csv, index=False)

    print(f"\n=== Conversion Summary ===")
    print(f"  Pure SMILES:              {counts['pure']}")
    print(f"  Wild labels kept:         {counts['wild']}")
    print(f"  Dropped (<r>/<c>/?):      {counts['drop_tag']}")
    print(f"  Failed (parse/validate):  {counts['fail']}")
    print(f"  Total output:             {len(df_out)}")
    print(f"  Saved to: {output_csv}")


if __name__ == '__main__':
    main()
