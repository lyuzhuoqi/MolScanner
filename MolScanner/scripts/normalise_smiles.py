#!/usr/bin/env python3
"""Normalise R-group representations in a USPTO molecule CSV.

Reads the MOL files referenced by the ``mol_path`` column to determine the
correct visual label for each wildcard atom (``*``) and rewrites SMILES
accordingly:

  - ``[N*]`` → ``[RN]``  (e.g. ``[1*]`` → ``[R1]``)
  - bare ``*`` → ``[<label>]`` where *<label>* comes from the **element
    symbol** in the MOL file's atom block (e.g. ``[X]``, ``[A]``, ``[M]``).
    Different bare ``*`` in the same molecule can receive different labels
    (per-atom mapping via RDKit canonical ordering).
    Falls back to ``[R]`` when the MOL file is unavailable, RDKit cannot
    parse it, or the label is not a valid R-group identifier.

Usage
-----
    python normalise_smiles.py \\
        --csv  data/uspto_mol/train_680k.csv \\
        --data_dir data \\
        --output data/uspto_mol/train_680k_normalised.csv
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------
def _is_valid_label(label: str) -> bool:
    """True if *label* is a usable R-group label for bracket encoding.

    Accepts alphanumeric strings of length 1-6 whose first character is
    uppercase.  This covers R, A, X, Q, L, M, R1a, R4O, Ar, Hal, etc.
    """
    return (
        1 <= len(label) <= 6
        and label[0].isupper()
        and label.isalnum()
    )


def _clean_element_symbol(sym: str) -> str:
    """Derive a clean R-group label from a MOL-file element symbol.

    Strips trailing non-alphanumeric characters (e.g. ``R5.`` → ``R5``).
    Maps ``*`` and ``R#`` to ``R``.
    """
    if sym in ('*', 'R#'):
        return 'R'
    cleaned = sym.rstrip('.,;:!+-*#')
    return cleaned if cleaned else 'R'


# ---------------------------------------------------------------------------
# MOL file parsing helpers
# ---------------------------------------------------------------------------
def parse_mol_element_symbols(mol_path: str) -> List[str]:
    """Read element symbols from the V2000 MOL file atom block.

    Returns a list of element symbols, one per atom, in MOL-file order
    (0-indexed).
    """
    try:
        with open(mol_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return []
    if len(lines) < 5:
        return []
    parts = lines[3].strip().split()
    try:
        n_atoms = int(parts[0])
    except (ValueError, IndexError):
        return []
    symbols = []
    for i in range(4, min(4 + n_atoms, len(lines))):
        p = lines[i].split()
        if len(p) >= 4:
            symbols.append(p[3])
    return symbols


# ---------------------------------------------------------------------------
# Per-atom mapping via RDKit
# ---------------------------------------------------------------------------
def _get_ordered_bare_labels(
    mol_path: str,
) -> Optional[List[str]]:
    """Determine the label for each bare ``*`` in canonical SMILES order.

    1. Parse the MOL file with RDKit to obtain the molecular graph.
    2. Identify non-numbered wildcard atoms (``atomic_num == 0``,
       ``isotope == 0``).
    3. Look up each atom's element symbol from the MOL-file text.
    4. Assign unique temporary isotopes, generate canonical SMILES, and
       read back the isotope order to establish the correspondence.

    Returns ``None`` on failure, or a list of label strings (one per bare
    ``*``, in the order they appear in canonical SMILES).
    """
    if not _HAS_RDKIT:
        return None

    elem_symbols = parse_mol_element_symbols(mol_path)
    if not elem_symbols:
        return None

    try:
        mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
    except Exception:
        return None
    if mol is None:
        return None

    # Collect bare-wildcard atom indices and their labels
    bare_info: Dict[int, str] = {}  # rdkit_idx → label
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 0:
            idx = atom.GetIdx()
            sym = elem_symbols[idx] if idx < len(elem_symbols) else '*'
            label = _clean_element_symbol(sym)
            if not _is_valid_label(label):
                label = 'R'
            bare_info[idx] = label

    if not bare_info:
        return []

    # Assign unique isotopes to determine canonical ordering
    rw = Chem.RWMol(mol)
    for idx in bare_info:
        rw.GetAtomWithIdx(idx).SetIsotope(1000 + idx)

    try:
        Chem.SanitizeMol(
            rw,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        temp_smiles = Chem.MolToSmiles(rw)
    except Exception:
        return None

    # Read isotopes in SMILES order → ordered labels
    # Only match isotopes >= 1000 (the temporary ones we assigned);
    # skip pre-existing numbered wildcards like [2*], [3*].
    ordered: List[str] = []
    for m in re.finditer(r'\[(\d+)\*\]', temp_smiles):
        iso = int(m.group(1))
        if iso < 1000:
            continue
        mol_idx = iso - 1000
        ordered.append(bare_info.get(mol_idx, 'R'))
    return ordered


# ---------------------------------------------------------------------------
# Annotation fragment filtering
# ---------------------------------------------------------------------------

# Fragments that are unambiguously text annotations, not chemistry.
_KNOWN_ANNOTATION_FRAGMENTS = frozenset([
    '[(]', '[)]', '[[]', '[]]', ']', '[',
    '[,]', '[;]', '[or]', '[and]',
    '[+]', '[-]', '[=]', '[:]',
    '[HH]', '[2HH]', '[3HH]',
])

# Regex for bracket-wrapped parenthesized labels: [(IX)], [(A)k], [(1)], etc.
_PAREN_LABEL_RE = re.compile(r'^\[\(.*\).*\]$')

# Regex for bare digit labels: [0], [1], [2], …
_BARE_DIGIT_RE = re.compile(r'^\[\d+\]$')

# Regex for bracket-wrapped lowercase single letters: [x], [i], [v], etc.
# These are OCR artefacts — real SMILES bracket atoms are uppercase.
_BRACKET_LOWERCASE_RE = re.compile(r'^\[[a-z]\]$')

# Regex for all-lowercase inside brackets: [ii], [xx], etc.
_BRACKET_MULTI_LOWER_RE = re.compile(r'^[a-z]{2,}$')

# Regex for OCR-artifact brackets with embedded ")": [)x], [)3], etc.
_BRACKET_PAREN_JUNK_RE = re.compile(r'^\[\).*\]$')

# Characters that are never valid inside a SMILES bracket atom.
_BRACKET_FORBIDDEN_RE = re.compile(r'[,?/!;]')

# Regex for Roman-numeral look-alikes as disconnected fragments.
# Matches fragments composed entirely of I (iodine), V (vanadium), X,
# brackets, parentheses, H, digits, charges — e.g. I[V](I)I → "(IV)".
_ROMAN_NUMERAL_RE = re.compile(r'^[IVX\[\]()H\d+\-]+$')

# Regex for embedded bare-digit atoms anywhere in a fragment: [0]I, I[3], etc.
# [0], [1], … are not valid SMILES atoms — they are digit OCR artefacts.
# Isotope notation is [\d+LETTER…] (e.g. [2H]) and won't match this pattern.
_EMBEDDED_BARE_DIGIT_RE = re.compile(r'\[\d+\]')


def _is_annotation_fragment(frag: str) -> bool:
    """Return True if *frag* is a text annotation rather than chemistry.

    Checks for common OCR artefacts found in patent MOL files: compound
    numbering (Roman numerals), stray parentheses, punctuation, etc.
    """
    if not frag:
        return True

    if frag in _KNOWN_ANNOTATION_FRAGMENTS:
        return True

    # Standalone bare digit(s): "2", "35" — not chemistry
    if frag.isdigit():
        return True

    # Unbalanced brackets: count of [ != count of ]
    if frag.count('[') != frag.count(']'):
        return True

    # Broken structure: ] appears before any [ — e.g. ]([, ]1[, ])=[
    first_close = frag.find(']')
    first_open = frag.find('[')
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        return True

    # Digit-bracket-digit splice: 1][4, 2][8, etc. — garbled OCR joins
    if re.search(r'\d\]\[\d', frag):
        return True

    # Single bracket-wrapped token: exactly one [ and one ]
    if (len(frag) >= 2 and frag[0] == '[' and frag[-1] == ']'
            and frag.count('[') == 1 and frag.count(']') == 1):
        inner = frag[1:-1]
        # [(IX)], [(A)k], [(1)], etc.
        if _PAREN_LABEL_RE.fullmatch(frag):
            return True
        # [0], [1], [2] — isolated digits (not [1*])
        if _BARE_DIGIT_RE.fullmatch(frag):
            return True
        # Truncated like [1), [2)
        if inner.endswith(')'):
            return True
        # Lowercase single letters: [x], [i], [v] — OCR artefacts
        if _BRACKET_LOWERCASE_RE.fullmatch(frag):
            return True
        # Multi-lowercase: [ii], [xx] — OCR artefacts
        if _BRACKET_MULTI_LOWER_RE.fullmatch(inner):
            return True
        # OCR junk with embedded ")": [)x], [)3], etc.
        if _BRACKET_PAREN_JUNK_RE.fullmatch(frag):
            return True
        # Forbidden characters inside bracket atom: , ? / ! ;
        if _BRACKET_FORBIDDEN_RE.search(inner):
            return True
        # Unmatched ( or ) inside bracket: [(X], [X)]
        if ('(' in inner) != (')' in inner):
            return True
        # No element symbol: no uppercase letter, no * — e.g. [2+], [+2], [-1]
        if not any(c.isupper() for c in inner) and '*' not in inner:
            return True
        # Leading charge before element: [-X], [+X] — invalid SMILES
        if len(inner) >= 2 and inner[0] in '-+' and inner[1].isalpha():
            return True
        # Trailing colon without class number: [X:] — invalid atom class
        if inner.endswith(':'):
            return True

    # Fragments containing bare-digit bracket atoms: [0]I, I[3], etc.
    # [\d+] (digit-only bracket) is never a valid SMILES atom.
    if _EMBEDDED_BARE_DIGIT_RE.search(frag) and frag != _EMBEDDED_BARE_DIGIT_RE.search(frag).group():
        return True

    # Roman-numeral look-alikes: disconnected fragments built entirely from
    # I, V, X and bracket/paren/H/charge characters.
    # Require ≥2 Roman letters so lone I (iodine) or [V] (vanadium) survive.
    if _ROMAN_NUMERAL_RE.fullmatch(frag):
        letters = re.sub(r'[\[\]()H\d+\-]', '', frag)
        if len(letters) >= 2 and all(c in 'IVX' for c in letters):
            return True

    return False


def remove_annotation_fragments(smiles: str) -> str:
    """Remove disconnected annotation fragments from a SMILES string.

    Splits on ``.``, drops fragments identified as text annotations
    (Roman numerals, stray parentheses, punctuation, etc.), and
    rejoins the remaining fragments.
    """
    if not isinstance(smiles, str) or '.' not in smiles:
        return smiles
    frags = smiles.split('.')
    kept = [f for f in frags if not _is_annotation_fragment(f)]
    return '.'.join(kept) if kept else smiles  # never return empty


# ---------------------------------------------------------------------------
# Bare-star counting / replacement helpers
# ---------------------------------------------------------------------------
def _count_bare_stars(smiles: str) -> int:
    """Count bare ``*`` (outside ``[…]`` brackets) in *smiles*."""
    count = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            j = smiles.find(']', i + 1)
            i = (j + 1) if j != -1 else (i + 1)
        elif smiles[i] == '*':
            count += 1
            i += 1
        else:
            i += 1
    return count


def _replace_bare_stars(smiles: str, labels: List[str]) -> str:
    """Replace each bare ``*`` in *smiles* with ``[label]`` from *labels*."""
    parts: List[str] = []
    idx = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            j = smiles.find(']', i + 1)
            if j == -1:
                j = len(smiles) - 1
            parts.append(smiles[i:j + 1])
            i = j + 1
        elif smiles[i] == '*':
            lab = labels[idx] if idx < len(labels) else 'R'
            parts.append(f'[{lab}]')
            idx += 1
            i += 1
        else:
            parts.append(smiles[i])
            i += 1
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------
def normalise_rgroup(
    smiles: str, mol_path: str = None, *, strip_annotations: bool = True,
) -> str:
    """Normalise wildcard atoms in *smiles* using the MOL file at *mol_path*.

    If *strip_annotations* is True (default), disconnected annotation
    fragments (Roman-numeral labels, stray parentheses, etc.) are removed
    before normalisation.

    Returns the normalised SMILES string.
    """
    if not isinstance(smiles, str):
        return smiles

    if strip_annotations:
        smiles = remove_annotation_fragments(smiles)

    # [N*] → [RN]  (numbered wildcard → named R-group)
    smiles = re.sub(r'\[(\d+)\*\]', r'[R\1]', smiles)
    if '*' not in smiles:
        return smiles

    bare_count = _count_bare_stars(smiles)
    if bare_count == 0:
        return smiles

    # Try per-atom mapping via RDKit + MOL element symbols
    if mol_path and os.path.isfile(mol_path):
        ordered = _get_ordered_bare_labels(mol_path)
        if ordered is not None and len(ordered) == bare_count:
            return _replace_bare_stars(smiles, ordered)

    # Fallback: replace all bare * with [R]
    return _replace_bare_stars(smiles, ['R'] * bare_count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Normalise R-group notation in a USPTO molecule CSV.')
    parser.add_argument('--csv', required=True,
                        help='Input CSV file (with file_path, mol_path, SMILES, …)')
    parser.add_argument('--data_dir', default='.',
                        help='Root directory for resolving relative mol_path entries')
    parser.add_argument('--output', required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    print(f'Reading {args.csv} …')
    df = pd.read_csv(args.csv)
    print(f'  {len(df)} rows, columns: {list(df.columns)}')

    has_mol_path = 'mol_path' in df.columns

    n_changed = 0
    new_smiles = []
    for idx in tqdm(range(len(df)), desc='Normalising'):
        row = df.iloc[idx]
        smiles = row['SMILES']
        mol_path = None
        if has_mol_path and isinstance(row['mol_path'], str):
            mol_path = os.path.join(args.data_dir, row['mol_path'])
        result = normalise_rgroup(smiles, mol_path)
        new_smiles.append(result)
        if result != smiles:
            n_changed += 1

    df['SMILES'] = new_smiles
    print(f'\nNormalised {n_changed}/{len(df)} rows '
          f'({100 * n_changed / len(df):.1f}%)')

    df.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
