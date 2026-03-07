"""
Convert MolParser E-SMILES to standard SMILES for RL fine-tuning.

E-SMILES format:
    backbone_SMILES<sep><a>atom_idx:label</a>...<r>atom_idx:rgroup</r>...

Design rationale — the GT SMILES produced here will be consumed by the RL
training loop (compute_reinforce_loss / compute_mrt_loss) which applies this
exact pipeline before computing any reward:

    s = remove_atom_mapping(s)
    s, mappings = _replace_functional_group(s)
    mol = Chem.MolFromSmiles(s, sanitize=False)
    s, mol = _expand_functional_group(mol, mappings)
    canon, ok = canonicalize_smiles(s, ignore_cistrans=True)

The rewards are: Tanimoto (Morgan FP), exact match, and visual (RDKit render
→ frozen-encoder cosine similarity).  All three require:
  • Chem.MolFromSmiles(canon) succeeds (full sanitization)
  • Chem.CanonSmiles works (deterministic canonical form)
  • Draw.MolToImage produces a meaningful rendering

Conversion rules:
  DROP entries whose tag section contains <r>, </r>, <c>, </c>, or ?
      — these encode structural relationships not representable in SMILES.
  ACCEPT all <a> labels (maximally inclusive):
    1. Pure SMILES (empty after <sep>): use backbone directly.
    2. <dum> (attachment point): keep as bare *.
    3. R-digit (R, R1, R2, ...): keep as [n*] isotope-labeled wildcard.
    4. RGROUP_SYMBOLS (X, Y, Z, Ar, ...): keep as bare *.
    5. ABBREVIATIONS (Boc, Ph, Me, ...): expand via _expand_functional_group.
    6. Anything else: try condensed-formula expansion via get_smiles_from_symbol;
       on failure the atom falls back to bare * (graceful degradation).
  Subscript brackets (R[1]→R1, CH[2]→CH2) and primes (R'→R) are normalised.
  Final SMILES is canonicalized with ignore_cistrans=True (matching training)
  and validated with full RDKit sanitization.

Usage:
    cd /path/to/Markush && python convert_esmiles.py
"""
import os
import sys
import re
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MolScanner'))
from rdkit import Chem
from constants import ABBREVIATIONS, RGROUP_SYMBOLS
from chemistry import _expand_functional_group, canonicalize_smiles

_PURE_RDIGIT_RE = re.compile(r'^R\d*$')
_SUBSCRIPT_RE = re.compile(r'\[(\w+)\]')   # R[1] → R1, CH[2] → CH2


def _normalize_label(label: str) -> str:
    """Strip E-SMILES subscript brackets and trailing primes."""
    label = _SUBSCRIPT_RE.sub(r'\1', label)
    label = label.rstrip("'")
    return label


def _validate_and_canonicalize(smiles: str, category: str):
    """Canonicalize with ignore_cistrans=False (although it will be ignored during evaluation)
    and validate with full RDKit sanitization."""
    canon, ok = canonicalize_smiles(smiles, ignore_cistrans=False)
    if not ok or not canon:
        return None, 'fail'
    if Chem.MolFromSmiles(canon) is None:
        return None, 'fail'
    return canon, category


def esmiles_to_smiles(esmiles: str):
    """Convert a single E-SMILES string to canonical SMILES.

    Returns (smiles | None, category) where category is one of:
    'pure', 'expanded', 'rgroup', 'drop_tag', 'fail'
    """
    if '<sep>' not in esmiles:
        return None, 'drop_tag'

    backbone, tags_str = esmiles.split('<sep>', 1)
    tags_str = tags_str.strip()

    # 1. Pure SMILES — empty after <sep>
    if not tags_str:
        return _validate_and_canonicalize(backbone, 'pure')

    # 2. Drop structural annotations not representable in SMILES
    if any(tok in tags_str for tok in ('<r>', '</r>', '<c>', '</c>', '?')):
        return None, 'drop_tag'

    # 3. Parse <a> tags
    a_tags = re.findall(r'<a>(\d+):(.+?)</a>', tags_str)
    if not a_tags:
        return _validate_and_canonicalize(backbone, 'pure')

    # 4. Build RDKit mol and annotate * atoms
    mol = Chem.MolFromSmiles(backbone, sanitize=False)
    if mol is None:
        return None, 'fail'

    mol_rw = Chem.RWMol(mol)
    has_rgroup = False
    has_expansion = False

    for idx_str, raw_label in a_tags:
        idx = int(idx_str)
        if idx >= mol_rw.GetNumAtoms():
            return None, 'fail'

        atom = mol_rw.GetAtomWithIdx(idx)
        if atom.GetSymbol() != '*':
            continue

        label = _normalize_label(raw_label)

        # (a) Attachment point
        if label == '<dum>' or label == 'dum':
            continue

        # (b) R-digit (R, R1, R2, ...)  →  [n*]
        if _PURE_RDIGIT_RE.match(label):
            has_rgroup = True
            digits = label[1:]
            if digits:
                atom.SetIsotope(int(digits))
            continue

        # (c) Other RGROUP_SYMBOLS (X, Y, Z, Q, A, E, Ar, Ra–Rd)  →  bare *
        if label in RGROUP_SYMBOLS:
            has_rgroup = True
            continue

        # (d) Everything else: set alias and let _expand_functional_group try
        #     (handles ABBREVIATIONS, condensed formulas; falls back to bare *)
        Chem.SetAtomAlias(atom, label)
        has_expansion = True

    # 5. Expand abbreviations / condensed formulas
    try:
        smiles, _ = _expand_functional_group(mol_rw, {})
    except Exception:
        return None, 'fail'

    # 6. Determine category
    if has_rgroup:
        cat = 'rgroup'
    elif has_expansion:
        cat = 'expanded'
    else:
        cat = 'pure'

    return _validate_and_canonicalize(smiles, cat)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'molparser_sft_real')
    output_csv = os.path.join(data_dir, 'sft_real.csv')

    print("Loading MolParser-7M sft_real from HuggingFace cache...")
    from datasets import load_dataset
    ds = load_dataset("UniParser/MolParser-7M", name="sft_real", split="train")
    print(f"Total entries: {len(ds)}")

    converted = []
    counts = {'pure': 0, 'expanded': 0, 'rgroup': 0, 'drop_tag': 0, 'fail': 0}

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
    print(f"  Expanded (abbrev/formula): {counts['expanded']}")
    print(f"  R-group (wildcard):       {counts['rgroup']}")
    print(f"  Dropped (<r>/<c>/?):      {counts['drop_tag']}")
    print(f"  Failed (parse/validate):  {counts['fail']}")
    print(f"  Total output:             {len(df_out)}")
    print(f"  Saved to: {output_csv}")


if __name__ == '__main__':
    main()
