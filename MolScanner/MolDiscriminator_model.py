import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import random
from tqdm import tqdm
import os
import socket
import tempfile
from func_timeout import FunctionTimedOut
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
import signal
import copy
import gc

from drawing_engine import (generate_image_from_smiles, generate_image_from_graph,
                            add_explicit_hydrogen, add_rgroup, add_rand_condensed, add_functional_group, 
                            _generate_style_config, _apply_style_config, 
                            _blank_image, _graph_to_molblock)
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ======================== Performance Optimizations ========================
# Enable cuDNN benchmark: safe because image_size is fixed (384x384)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Use TensorFloat-32 for faster matmul on Ampere+ GPUs
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
# Disable cv2 threading to avoid conflicts with DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Use 'forkserver' to avoid segfaults from Indigo C library in forked workers
_DL_MP_CONTEXT = 'forkserver'

# Maximum number of DataLoader crash retries per epoch before giving up
_MAX_WORKER_RETRIES = 3


def _worker_init_fn(worker_id: int):
    """Re-initialize per-worker state to avoid issues from forked C libraries."""
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    # Re-seed so each worker produces different augmentations
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    """Create a warmup + cosine-annealing LR schedule function.

    Args:
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (warmup + cosine phase).

    Returns:
        A ``lr_lambda`` callable for ``torch.optim.lr_scheduler.LambdaLR``.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return lr_lambda


def _compute_curriculum_ramp(
    curriculum_epochs: Optional[int],
    num_epochs: int,
) -> int:
    """Compute the curriculum ramp length (number of epoch *intervals*).

    During the ramp the hard-negative probability is linearly increased
    from ``hard_negative_min_prob`` to ``hard_negative_max_prob``.

    Args:
        curriculum_epochs: Explicit ramp length, or ``None`` for the
            default of ``num_epochs // 2``.
        num_epochs: Total training epochs.

    Returns:
        Positive integer used as the denominator in the linear ramp.
    """
    raw = (curriculum_epochs if curriculum_epochs is not None
           else max(1, num_epochs // 2)) - 1
    return max(1, raw)


def _resolve_tb_log_dir(
    save_path: Optional[str],
    resume: bool,
) -> str:
    """Determine the TensorBoard log directory.

    On fresh runs a new timestamped sub-directory is created under
    ``<save_path>/logs/``.  When *resume* is ``True`` the most recent
    existing log directory is reused so that TensorBoard curves stay
    continuous across restarts.

    Args:
        save_path: Root checkpoint / log directory (may be ``None``).
        resume: Whether we are resuming from a previous run.

    Returns:
        Absolute path to the TensorBoard log directory.
    """
    if resume and save_path:
        logs_root = os.path.join(save_path, 'logs')
        if os.path.isdir(logs_root):
            existing = sorted(os.listdir(logs_root))
            if existing:
                return os.path.join(logs_root, existing[-1])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_path:
        return os.path.join(save_path, 'logs', timestamp)
    return f'./runs/{timestamp}'

# ================ Hard Negative Generation Probabilities =================
# Structural perturbation probabilities (RDKit-based with sanitization)
HARD_NEG_DROP_ATOM_PROB = 0.8     # Probability of dropping 1-3 leaf atoms
HARD_NEG_ADD_ATOM_PROB = 0.8      # Probability of attaching a random atom
HARD_NEG_CHANGE_BOND_PROB = 0.8   # Probability of changing bond order / chirality
HARD_NEG_CHANGE_SYMBOL_PROB = 0.8 # Probability of swapping an atom symbol
HARD_NEG_BREAK_BOND_PROB = 0.3    # Probability of breaking a bond (keep all fragments)
# Indigo augmentation probabilities (applied after structural perturbation)
HARD_NEG_HYDROGEN_PROB = 0.3      # Probability of adding explicit hydrogens
HARD_NEG_RGROUP_PROB = 0.3        # Probability of adding an R-group
HARD_NEG_CONDENSED_PROB = 0.2     # Probability of adding a condensed formula
HARD_NEG_FUNCTIONAL_GROUP_PROB = 0.3  # Probability of replacing functional groups


class PadToSquare:
    """
    Simple callable that pads an image to be square and adjusts keypoints.
    Not an Albumentations transform - used separately.
    """
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, image, keypoints=None):
        h, w = image.shape[:2]
        target_dim = max(h, w)
        
        pad_h = target_dim - h
        pad_w = target_dim - w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Handle fill value for color images - convert to tuple of floats
        if image.ndim == 3 and image.shape[2] == 3:
            fill_value = (float(self.fill), float(self.fill), float(self.fill))
        else:
            fill_value = (float(self.fill),)
        
        padded_img = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=fill_value
        )
        
        # Adjust keypoints if provided
        if keypoints is not None:
            adjusted_kps = []
            for kp in keypoints:
                x, y = kp[0], kp[1]
                adjusted_kps.append([x + left, y + top])
            return padded_img, adjusted_kps
        
        return padded_img, None
    

def _perturb_bond_rdkit(
    mol: Chem.RWMol,
    debug: bool = False,
) -> Chem.RWMol:
    """Safely perturb a bond in an RDKit RWMol with sanitization.

    Randomly picks one of two operation categories (50/50 when both
    are available):

      - **change order** – toggle a single bond to double or vice-versa.
        Candidates are shuffled and tried one by one; a valence
        pre-filter skips bonds where upgrading to double would exceed
        the maximum valence of either endpoint.  The first candidate
        that passes ``Chem.SanitizeMol`` is accepted.
      - **flip stereo** – flip the stereo configuration of a double
        bond (E/Z) or a wedge/dash bond direction.

    On sanitization failure the change is rolled back and (for order
    changes) the next candidate is tried.  If all candidates fail the
    molecule is returned unchanged.

    Args:
        mol: an RDKit ``RWMol`` (modified **in-place** on success).
        debug: print detailed information about what was attempted.

    Returns:
        The (possibly mutated) molecule.
    """
    bonds = list(mol.GetBonds())
    if not bonds:
        return mol

    # Classify bonds
    order_candidates = []  # (bond_idx, current_type)
    stereo_candidates = []  # (bond_idx, kind)  kind='double_stereo' | 'wedge'
    for bond in bonds:
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            order_candidates.append((bond.GetIdx(), bt))
        elif bt == Chem.BondType.DOUBLE:
            order_candidates.append((bond.GetIdx(), bt))
            if bond.GetStereo() not in (Chem.BondStereo.STEREONONE, None):
                stereo_candidates.append((bond.GetIdx(), 'double_stereo'))
        # Wedge / dash single bonds
        bd = bond.GetBondDir()
        if bd in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            stereo_candidates.append((bond.GetIdx(), 'wedge'))

    if not order_candidates and not stereo_candidates:
        return mol

    # Decide operation: 50% order change, 50% stereo flip (if available)
    do_stereo = stereo_candidates and (random.random() < 0.5 or not order_candidates)

    if do_stereo:
        bond_idx, kind = random.choice(stereo_candidates)
        bond = mol.GetBondWithIdx(bond_idx)
        old_dir = bond.GetBondDir()
        old_stereo = bond.GetStereo()
        if kind == 'wedge':
            new_dir = (Chem.BondDir.BEGINDASH
                       if old_dir == Chem.BondDir.BEGINWEDGE
                       else Chem.BondDir.BEGINWEDGE)
            bond.SetBondDir(new_dir)
            if debug:
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                print(f'  [perturb] flip wedge: bond {a1}-{a2} '
                      f'{old_dir.name}->{new_dir.name}')
        else:  # double_stereo
            flip_map = {
                Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
                Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
                Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS,
            }
            new_stereo = flip_map.get(old_stereo, old_stereo)
            bond.SetStereo(new_stereo)
            if debug:
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                print(f'  [perturb] flip double-bond stereo: bond {a1}-{a2} '
                      f'{old_stereo.name}->{new_stereo.name}')
        # Stereo flips don't change valence, but validate anyway
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Roll back
            bond = mol.GetBondWithIdx(bond_idx)
            if kind == 'wedge':
                bond.SetBondDir(old_dir)
            else:
                bond.SetStereo(old_stereo)
            if debug:
                print('  [perturb] stereo flip failed sanitization, rolled back')
        return mol

    # ---- Order change (try multiple candidates until one succeeds) ----
    random.shuffle(order_candidates)
    for bond_idx, old_bt in order_candidates:
        bond = mol.GetBondWithIdx(bond_idx)
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        new_bt = Chem.BondType.DOUBLE if old_bt == Chem.BondType.SINGLE else Chem.BondType.SINGLE

        # Pre-filter: if upgrading to double, both atoms need room for one
        # more bond (i.e. current_val < max_val).  Skip obvious failures.
        if new_bt == Chem.BondType.DOUBLE:
            skip = False
            for ai in (a1, a2):
                atom = mol.GetAtomWithIdx(ai)
                try:
                    max_val = max(Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum()))
                except Exception:
                    max_val = 4
                if atom.GetExplicitValence() >= max_val:
                    skip = True
                    break
            if skip:
                if debug:
                    print(f'  [perturb] change_bond_order: skipping bond {a1}-{a2} '
                          f'(valence pre-filter)')
                continue

        bond.SetBondType(new_bt)
        # Clear stereo info when changing order to avoid inconsistencies
        old_dir = bond.GetBondDir()
        old_stereo_info = bond.GetStereo()
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)

        try:
            Chem.SanitizeMol(mol)
            if debug:
                print(f'  [perturb] change_bond_order: bond {a1}-{a2} '
                      f'{old_bt.name}->{new_bt.name}')
            return mol
        except Exception:
            # Roll back this bond and try the next candidate
            bond = mol.GetBondWithIdx(bond_idx)
            bond.SetBondType(old_bt)
            bond.SetBondDir(old_dir)
            bond.SetStereo(old_stereo_info)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass  # original should always be valid
            if debug:
                print(f'  [perturb] change_bond_order FAILED (valence), rolled back '
                      f'bond {a1}-{a2}')
            continue

    if debug:
        print('  [perturb] change_bond_order: no valid order change found for any bond')
    return mol


def _perturb_symbol_rdkit(
    mol: Chem.RWMol,
    debug: bool = False,
) -> Chem.RWMol:
    """Safely swap an atom symbol in an RDKit RWMol with sanitization.

    Picks a random non-wildcard heavy atom and replaces it with a
    different common atom (C, N, O, F, Cl, Br, S).  After swapping the
    molecule is sanitized; if sanitization fails (valence error, etc.)
    the change is rolled back and the original molecule is returned.

    The function uses valence-aware filtering: candidate replacement
    atoms are restricted to those whose RDKit default valence is >=
    the current degree of the target atom, which avoids most valence
    errors upfront while still allowing the sanitization fallback for
    edge cases.

    Args:
        mol: an RDKit ``RWMol`` (modified **in-place** on success).
        debug: print detailed information about what was attempted.

    Returns:
        The (possibly mutated) molecule.
    """
    COMMON_ATOMS = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']
    # Map symbol -> set of typical valences (used for pre-filtering)
    VALENCE_MAP = {
        'C': {4}, 'N': {3, 5}, 'O': {2}, 'F': {1},
        'Cl': {1}, 'Br': {1}, 'S': {2, 4, 6},
    }

    # Collect candidate atoms (non-wildcard heavy atoms)
    atom_indices = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != '*' and sym in COMMON_ATOMS:
            atom_indices.append(atom.GetIdx())

    if not atom_indices:
        return mol

    random.shuffle(atom_indices)

    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        current_sym = atom.GetSymbol()
        degree = atom.GetDegree()  # number of explicit bonds

        # Filter candidates by valence compatibility
        candidates = []
        for sym in COMMON_ATOMS:
            if sym == current_sym:
                continue
            max_val = max(VALENCE_MAP.get(sym, {4}))
            if max_val >= degree:
                candidates.append(sym)

        if not candidates:
            continue

        new_sym = random.choice(candidates)

        # Save old state
        old_atomic_num = atom.GetAtomicNum()
        old_formal_charge = atom.GetFormalCharge()
        old_num_explicit_hs = atom.GetNumExplicitHs()
        old_no_implicit = atom.GetNoImplicit()

        # Apply change
        new_atom_num = Chem.GetPeriodicTable().GetAtomicNumber(new_sym)
        atom.SetAtomicNum(new_atom_num)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)

        try:
            Chem.SanitizeMol(mol)
            if debug:
                print(f'  [perturb] change_symbol: atom {atom_idx} '
                      f'{current_sym}->{new_sym}')
            return mol
        except Exception:
            # Roll back
            atom = mol.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(old_atomic_num)
            atom.SetFormalCharge(old_formal_charge)
            atom.SetNumExplicitHs(old_num_explicit_hs)
            atom.SetNoImplicit(old_no_implicit)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            if debug:
                print(f'  [perturb] change_symbol FAILED (valence): atom {atom_idx} '
                      f'{current_sym}->{new_sym}, rolled back')
            continue  # Try next atom

    if debug:
        print('  [perturb] change_symbol: no valid swap found for any atom')
    return mol


def _drop_atom_rdkit(
    mol: Chem.RWMol,
    debug: bool = False,
) -> Chem.RWMol:
    """Safely remove 1-3 leaf atoms from an RDKit RWMol with sanitization.

    A *leaf atom* is any atom with exactly one bond (degree == 1),
    including explicit hydrogen atoms.  The function removes 1-3
    randomly chosen leaf atoms, then sanitizes.  If sanitization fails,
    all removals are rolled back.

    Args:
        mol: an RDKit ``RWMol``.
        debug: print information about the operation.

    Returns:
        The (possibly smaller) molecule.
    """
    # Save original in case we need to roll back
    orig_block = Chem.MolToMolBlock(mol)

    # Identify leaf atoms (degree == 1, including H, excluding wildcard *)
    leaf_indices = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1 and atom.GetSymbol() != '*':
            leaf_indices.append(atom.GetIdx())

    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    n_atoms = mol.GetNumAtoms()
    if not leaf_indices or n_heavy <= 3:
        if debug:
            print('  [perturb] drop_atom: no eligible leaf atoms or molecule too small')
        return mol

    # Ensure we keep at least 3 heavy atoms after removal
    max_drop = min(3, len(leaf_indices), n_atoms - 3)
    num_drop = random.randint(1, max(1, max_drop))
    to_drop = sorted(random.sample(leaf_indices, num_drop), reverse=True)

    # Check that we won't remove too many heavy atoms
    heavy_drop = sum(1 for i in to_drop if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
    if n_heavy - heavy_drop < 3:
        # Reduce to only drop H atoms, or fewer heavy atoms
        h_only = [i for i in to_drop if mol.GetAtomWithIdx(i).GetAtomicNum() == 1]
        heavy_only = [i for i in to_drop if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        allowed_heavy = max(0, n_heavy - 3)
        to_drop = sorted(h_only + heavy_only[:allowed_heavy], reverse=True)
        if not to_drop:
            if debug:
                print('  [perturb] drop_atom: cannot drop without going below 3 heavy atoms')
            return mol

    if debug:
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in to_drop]
        print(f'  [perturb] drop_atom: removing {len(to_drop)} leaf atom(s) '
              f'at indices {to_drop} ({syms})')

    for idx in to_drop:
        mol.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        if debug:
            print('  [perturb] drop_atom FAILED sanitization, rolled back')
        # Roll back: re-parse original
        rd_mol = Chem.MolFromMolBlock(orig_block, sanitize=True)
        if rd_mol is not None:
            return Chem.RWMol(rd_mol)
        return mol  # should not happen


def _add_atom_rdkit(
    mol: Chem.RWMol,
    debug: bool = False,
) -> Chem.RWMol:
    """Safely attach a random common atom to the molecule with sanitization.

    Picks a random existing non-wildcard atom as the parent and attaches
    a new atom (from C, N, O, F, Cl, Br, S) via a single bond.  Only
    atoms whose current valence allows an additional single bond are
    considered as parents.  After addition the molecule is sanitized;
    if it fails the addition is rolled back.

    Args:
        mol: an RDKit ``RWMol``.
        debug: print information about the operation.

    Returns:
        The (possibly larger) molecule.
    """
    COMMON_ATOMS = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']

    # Save original for rollback
    orig_block = Chem.MolToMolBlock(mol)

    # Find candidate parent atoms that have room for another single bond
    parent_candidates = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            continue
        # Check if atom can accept one more bond
        try:
            max_val = max(Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum()))
        except Exception:
            max_val = 4
        current_val = atom.GetExplicitValence()
        if current_val < max_val:
            parent_candidates.append(atom.GetIdx())

    if not parent_candidates:
        if debug:
            print('  [perturb] add_atom: no parent atom with available valence')
        return mol

    parent_idx = random.choice(parent_candidates)
    new_sym = random.choice(COMMON_ATOMS)
    new_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_sym)

    if debug:
        parent_sym = mol.GetAtomWithIdx(parent_idx).GetSymbol()
        print(f'  [perturb] add_atom: {new_sym} attached to '
              f'atom {parent_idx} ({parent_sym})')

    new_idx = mol.AddAtom(Chem.Atom(new_atomic_num))
    mol.AddBond(parent_idx, new_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        if debug:
            print('  [perturb] add_atom FAILED sanitization, rolled back')
        rd_mol = Chem.MolFromMolBlock(orig_block, sanitize=True)
        if rd_mol is not None:
            return Chem.RWMol(rd_mol)
        return mol


def _break_molecule_rdkit(
    mol: Chem.RWMol,
    debug: bool = False,
) -> Chem.RWMol:
    """Randomly break a bond to split the molecule, keeping all fragments.

    Picks a random non-ring single bond (to avoid destroying ring systems)
    and removes it.  The resulting disconnected molecule (with all
    fragments) is returned as-is — Indigo will render them side by side.
    The molecule is sanitized after the operation; on failure the
    original molecule is returned.

    Args:
        mol: an RDKit ``RWMol``.
        debug: print information about the operation.

    Returns:
        The molecule with the bond removed (may contain multiple
        disconnected fragments), or the original molecule if no suitable
        bond was found or sanitization failed.
    """
    orig_block = Chem.MolToMolBlock(mol)

    # Find candidate bonds: non-ring single bonds between heavy atoms
    candidates = []
    ri = mol.GetRingInfo()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if ri.NumBondRings(bond.GetIdx()) > 0:
            continue  # skip ring bonds — breaking them creates weird valences
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetSymbol() == '*' or a2.GetSymbol() == '*':
            continue
        candidates.append(bond.GetIdx())

    if not candidates:
        if debug:
            print('  [perturb] break_molecule: no eligible non-ring single bond')
        return mol

    bond_idx = random.choice(candidates)
    bond = mol.GetBondWithIdx(bond_idx)
    a1_idx, a2_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

    if debug:
        a1_sym = mol.GetAtomWithIdx(a1_idx).GetSymbol()
        a2_sym = mol.GetAtomWithIdx(a2_idx).GetSymbol()
        print(f'  [perturb] break_molecule: breaking bond {a1_idx}({a1_sym})'
              f'-{a2_idx}({a2_sym})')

    mol.RemoveBond(a1_idx, a2_idx)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        if debug:
            print('  [perturb] break_molecule FAILED sanitization after bond removal, rolled back')
        rd_mol = Chem.MolFromMolBlock(orig_block, sanitize=True)
        if rd_mol is not None:
            return Chem.RWMol(rd_mol)
        return mol

    if debug:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        frag_info = [(f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)) for f in frags] if frags else []
        print(f'  [perturb] break_molecule: {len(frag_info)} fragment(s): {frag_info}')

    return mol


def _generate_hard_negative_from_graph(
    graph: Dict[str, Any],
    default_drawing_style: bool = False,
    debug: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Generate a hard negative image from a molecule graph.

    This is a unified pipeline that first applies **RDKit-based structural
    perturbations** (with sanitization and rollback) and then **Indigo
    augmentations**.  Each operation is governed by its own probability
    constant (``HARD_NEG_*`` at module level), so they fire independently
    and can stack.

    **Structural perturbations** (RDKit, applied to molblock):
      - *drop_atom* – remove 1-3 leaf atoms including H
        (prob ``HARD_NEG_DROP_ATOM_PROB``)
      - *add_atom* – attach a random common atom
        (prob ``HARD_NEG_ADD_ATOM_PROB``)
      - *change_bond* – toggle bond order or flip wedge/stereo;
        tries multiple candidates (prob ``HARD_NEG_CHANGE_BOND_PROB``)
      - *change_symbol* – swap an atom symbol with valence pre-filtering
        (prob ``HARD_NEG_CHANGE_SYMBOL_PROB``)
      - *break_molecule* – break a non-ring single bond, keep all
        fragments (prob ``HARD_NEG_BREAK_BOND_PROB``)

    **Indigo augmentations** (after converting to Indigo molecule):
      - *add_explicit_hydrogen* (prob ``HARD_NEG_HYDROGEN_PROB``)
      - *add_rgroup* (prob ``HARD_NEG_RGROUP_PROB``)
      - *add_rand_condensed* (prob ``HARD_NEG_CONDENSED_PROB``)
      - *add_functional_group* (prob ``HARD_NEG_FUNCTIONAL_GROUP_PROB``)

    Returns:
        ``(image, success)`` – *image* is an (H, W, 3) uint8 ndarray.
    """
    from indigo import Indigo
    from indigo.renderer import IndigoRenderer

    graph = copy.deepcopy(graph)
    applied_ops: List[str] = []

    # ------------------------------------------------------------------ #
    #  RDKit perturbation, Indigo augmentation & render                   #
    # ------------------------------------------------------------------ #
    try:
        mol_block = _graph_to_molblock(graph)

        # -- Structural perturbations (RDKit-based with sanitization) --
        do_drop_atom = random.random() < HARD_NEG_DROP_ATOM_PROB
        do_add_atom = random.random() < HARD_NEG_ADD_ATOM_PROB
        do_change_bond = random.random() < HARD_NEG_CHANGE_BOND_PROB
        do_change_symbol = random.random() < HARD_NEG_CHANGE_SYMBOL_PROB
        do_break_bond = random.random() < HARD_NEG_BREAK_BOND_PROB
        if do_drop_atom or do_add_atom or do_change_bond or do_change_symbol or do_break_bond:
            rd_mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
            if rd_mol is not None:
                rw_mol = Chem.RWMol(rd_mol)
                if do_drop_atom:
                    rw_mol = _drop_atom_rdkit(rw_mol, debug=debug)
                    applied_ops.append('drop_atom')
                if do_add_atom:
                    rw_mol = _add_atom_rdkit(rw_mol, debug=debug)
                    applied_ops.append('add_atom')
                if do_change_bond:
                    rw_mol = _perturb_bond_rdkit(rw_mol, debug=debug)
                    applied_ops.append('change_bond')
                if do_change_symbol:
                    rw_mol = _perturb_symbol_rdkit(rw_mol, debug=debug)
                    applied_ops.append('change_symbol')
                if do_break_bond:
                    rw_mol = _break_molecule_rdkit(rw_mol, debug=debug)
                    applied_ops.append('break_molecule')
                mol_block = Chem.MolToMolBlock(rw_mol)
            elif debug:
                print('  [perturb] RDKit perturbations skipped: '
                      'could not parse molblock')

        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        ind_mol = indigo.loadMolecule(mol_block)
        smiles = ind_mol.canonicalSmiles()
        if debug:
            print(f'  [indigo_aug] input SMILES: {smiles}')

        if random.random() < HARD_NEG_HYDROGEN_PROB:
            ind_mol = add_explicit_hydrogen(ind_mol)
            applied_ops.append('add_explicit_hydrogen')
        if random.random() < HARD_NEG_RGROUP_PROB:
            ind_mol = add_rgroup(ind_mol, smiles)
            applied_ops.append('add_rgroup')
        if random.random() < HARD_NEG_CONDENSED_PROB:
            ind_mol = add_rand_condensed(ind_mol)
            applied_ops.append('add_rand_condensed')
        if random.random() < HARD_NEG_FUNCTIONAL_GROUP_PROB:
            ind_mol = add_functional_group(indigo, ind_mol)
            applied_ops.append('add_functional_group')

        if debug:
            print(f'  [hard_neg] applied ops: {applied_ops}')

        # Style & render
        ind_mol, style_config, mol_config = _generate_style_config(
            ind_mol, default_drawing_style)
        _apply_style_config(indigo, style_config, ind_mol, mol_config)
        ind_mol.layout()
        buf = renderer.renderToBuffer(ind_mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
        if debug:
            print(f'  [hard_neg] render OK, shape={img.shape}')
        return img, True
    except Exception as e:
        if debug:
            print(f'  [hard_neg] render FAILED: {e}')
        return _blank_image(), False


class TripletMoleculeDataset(Dataset):
    """Dataset for generating triplets (anchor, positive, negative) of molecule images."""
    

    def __init__(
        self, 
        smiles_list: List[str],
        image_size: Tuple[int, int] = (384, 384),
        mol_augment: bool = True,
        img_augment: bool = True,
        geo_augment: bool = True,
        default_drawing_style: bool = False,
        shuffle_smiles: bool = True,
        hard_negative_prob: float = 0.0,
        debug: bool = False,
    ):
        """
        Args:
            smiles_list: List of SMILES strings
            image_size: Target image size (height, width)
            mol_augment: Whether to apply molecule augmentation
            img_augment: Whether to apply image augmentation
            geo_augment: Whether to apply geometric augmentation
            default_drawing_style: Whether to use default drawing style
            shuffle_smiles: Whether to shuffle the SMILES list at initialization
            hard_negative_prob: Probability of generating a hard negative (same
                molecule, different augmentation) instead of a random different
                molecule. Updated per epoch for curriculum learning.
            debug: Print detailed information about modifications/augmentations
                applied during hard negative sample generation.
        """
        self.smiles_list = smiles_list
        self.hard_negative_prob = hard_negative_prob
        self.debug = debug
        if shuffle_smiles:
            print('Shuffling SMILES list for dataset initialization...')
            rng = np.random.default_rng(42)
            shuffled_indices = rng.permutation(len(self.smiles_list))
            self.smiles_list = [self.smiles_list[i] for i in shuffled_indices]
            print('SMILES list shuffled.')
        self.image_size = image_size
        self.mol_augment = mol_augment
        self.img_augment = img_augment
        self.geo_augment = geo_augment
        self.default_drawing_style = default_drawing_style

        # Separate pad to square operation (handles keypoints manually)
        self.pad_to_square = PadToSquare(fill=255)
        
        # 1. Rotation with fit_output (Albumentations handles keypoints)
        self.geo_transforms_list = []
        if self.geo_augment:
            self.geo_transforms_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=1.0),
                A.RandomCropFromBorders(crop_left=0.01, crop_right=0.01, crop_top=0.01, crop_bottom=0.01, p=0.5),
                A.CropAndPad(percent=(0.0, 0.4), sample_independently=True, keep_size=False, fill=255, fill_mask=255, p=0.2),
            ]
        self.geo_transforms = A.Compose(self.geo_transforms_list)
        # 2. Blur and noise augmentation
        self.img_transforms_list = []
        if self.img_augment:
            self.img_transforms_list += [
                A.Downscale(scale_range=(0.5, 0.8), interpolation_pair={'upscale': 3, 'downscale': 3}),
                A.Blur(),
                A.GaussNoise(),
                A.SaltAndPepper()
            ]
        self.img_transforms = A.Compose(self.img_transforms_list)
        # 3. Final transforms (after padding, no keypoints needed)
        self.post_transforms = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_AREA),
            A.ToGray(num_output_channels=3),  # Keep 3 channels for pretrained backbone
            A.Normalize(),  # ImageNet normalization
            ToTensorV2(),
        ])
        

    def __len__(self) -> int:
        return len(self.smiles_list)

    # ---- Helper methods for triplet generation ----

    def _try_generate_from_smiles(
        self, smiles: str, mol_augment: bool
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """Try generating an image + graph from SMILES. Returns None on failure."""
        try:
            img, _, graph, success, _, _ = generate_image_from_smiles(
                smiles, mol_augment=mol_augment,
                default_drawing_style=self.default_drawing_style, debug=False)
            if success:
                return img, graph
        except (FunctionTimedOut, Exception):
            pass
        return None

    def _try_graph_negative(self, graph: Dict[str, Any]) -> Optional[np.ndarray]:
        """Try generating a hard negative from the graph via the unified pipeline."""
        if self.debug:
            print('[hard_neg] strategy=graph_perturb')
        img, success = _generate_hard_negative_from_graph(
            graph, default_drawing_style=self.default_drawing_style,
            debug=self.debug)
        return img if success else None

    def _generate_negative_image(
        self, anchor_smiles: str, anchor_idx: int, graph: Dict[str, Any]
    ) -> np.ndarray:
        """Generate a negative sample using curriculum-based strategy selection.

        With probability ``hard_negative_prob`` a *hard* negative is generated
        (same molecule re-augmented, or graph structurally perturbed / Indigo-
        augmented).  Otherwise an *easy* negative from a different molecule is
        used.  Falls back gracefully through strategies on failure.
        """
        if random.random() < self.hard_negative_prob:
            strategy = random.choice(['smiles_reaugment', 'graph_perturb'])
            if self.debug:
                print(f'[hard_neg] idx={anchor_idx}, smiles={anchor_smiles[:60]}, '
                      f'strategy={strategy}')
            # Try graph-based hard negative when selected
            if strategy == 'graph_perturb' and graph:
                img = self._try_graph_negative(graph)
                if img is not None:
                    return img
                if self.debug:
                    print('  [hard_neg] graph_perturb failed, falling back to smiles_reaugment')
            # SMILES re-augmentation (primary strategy or graph fallback)
            if self.debug and strategy == 'smiles_reaugment':
                print('[hard_neg] strategy=smiles_reaugment (re-augmenting anchor SMILES)')
            result = self._try_generate_from_smiles(anchor_smiles, mol_augment=True)
            if result is not None:
                if self.debug:
                    print(f'  [hard_neg] smiles_reaugment OK, shape={result[0].shape}')
                return result[0]
            if self.debug:
                print('  [hard_neg] smiles_reaugment FAILED, falling back to easy negative')

        # Easy negative: use a different molecule
        while True:
            neg_idx = random.randint(0, len(self.smiles_list) - 1)
            if neg_idx != anchor_idx:
                break
        if self.debug:
            print(f'[easy_neg] idx={anchor_idx}, using different molecule '
                  f'neg_idx={neg_idx}, smiles={self.smiles_list[neg_idx][:60]}')
        result = self._try_generate_from_smiles(
            self.smiles_list[neg_idx], mol_augment=self.mol_augment)
        return result[0] if result is not None else _blank_image()

    def _apply_transforms(self, img: np.ndarray) -> torch.Tensor:
        """Apply geo, pad, image, and post transforms to a single image."""
        img = self.geo_transforms(image=img)['image']
        img, _ = self.pad_to_square(img)
        img = self.img_transforms(image=img)['image']
        return self.post_transforms(image=img)['image']

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a triplet (anchor, positive, negative)."""
        anchor_smiles = self.smiles_list[idx]

        # Generate positive (with molecule augmentation)
        result = self._try_generate_from_smiles(anchor_smiles, mol_augment=self.mol_augment)
        if result is not None:
            positive_img, graph = result
            # Generate anchor from graph (cycle consistency)
            try:
                anchor_img, _, _, success = generate_image_from_graph(
                    graph, default_drawing_style=self.default_drawing_style)
                if not success:
                    anchor_img = positive_img
            except (FunctionTimedOut, Exception):
                anchor_img = positive_img
            # Generate negative
            negative_img = self._generate_negative_image(anchor_smiles, idx, graph)
        else:
            positive_img = _blank_image()
            anchor_img = positive_img
            negative_img = _blank_image()

        return (
            self._apply_transforms(anchor_img),
            self._apply_transforms(positive_img),
            self._apply_transforms(negative_img),
        )


class SiameseNetwork(nn.Module):
    """
    Siamese network with EfficientNet_V2_S backbone for embedding molecule images.
    """
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_v2_s(weights=weights)
        feat_dim = efficientnet.classifier[1].in_features  # 1280
        efficientnet.classifier = nn.Identity()
        self.backbone = efficientnet
        self.proj = nn.Linear(feat_dim, embedding_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.proj(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)


class TripletLoss(nn.Module):
    """Triplet loss for Siamese network training."""
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Loss = max(d(a, p) - d(a, n) + margin, 0)
        where d is the Euclidean distance.
        """
        # Euclidean distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class _EmbeddingWrapper(nn.Module):
    """Wrapper so that nn.DataParallel calls forward_once via forward()."""
    def __init__(self, siamese_model: SiameseNetwork):
        super().__init__()
        self.siamese_model = siamese_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.siamese_model.forward_once(x)


def _find_free_port() -> int:
    """Find a free port on localhost for DDP."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _ddp_train_worker(rank: int, world_size: int, config: Dict[str, Any]):
    """
    Per-process DDP training worker launched by mp.spawn.
    Each rank creates its own model, optimizer, and data loader.
    """
    # ---- DDP setup ----
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(config['master_port'])
    # Make NCCL more tolerant: allow async error handling and increase timeout
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    nccl_timeout = timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=nccl_timeout)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    is_main = (rank == 0)

    # Register signal handlers so SIGTERM/SIGINT trigger cleanup
    def _graceful_exit(signum, frame):
        raise SystemExit(f'Rank {rank} received signal {signum}')
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    writer = None  # declare early so finally block can close it
    try:  # ensure dist.destroy_process_group() is always called
        _ddp_train_worker_body(rank, world_size, config, device, is_main)
    except Exception as e:
        if is_main:
            print(f'[Rank {rank}] Training error: {e}')
        raise
    finally:
        # Force-release all GPU memory held by this process
        torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.destroy_process_group()
        # Clear CUDA caches and reset device to release all allocations
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()


def _ddp_train_worker_body(rank: int, world_size: int, config: Dict[str, Any],
                           device: torch.device, is_main: bool):
    """Core DDP training loop executed by each worker spawned via ``mp.spawn``.

    Handles model construction, dataset creation, optimizer / scheduler setup,
    optional checkpoint resume, the full train-validate-checkpoint epoch loop
    with DataLoader crash retry, curriculum learning for hard-negative ramp,
    and early stopping broadcast across ranks.

    Separated from :func:`_ddp_train_worker` so that the caller can wrap it
    in a ``try / finally`` to guarantee ``dist.destroy_process_group()``.
    """
    # ---- Reproducibility (per-rank seed diversity for augmentation) ----
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    # ---- Unpack frequently-used config values once ----
    num_epochs: int        = config['num_epochs']
    batch_size: int        = config['batch_size']
    num_workers: int       = config['num_workers']
    save_path: str         = config['save_path']
    start_epoch: int       = config.get('resume_epoch', 0)
    use_amp: bool          = config.get('use_amp', True)
    warmup_ratio: float    = config.get('warmup_ratio', 0.02)
    hard_neg_max: float    = config.get('hard_negative_max_prob', 0.0)
    hard_neg_min: float    = config.get('hard_negative_min_prob', 0.0)
    curriculum_epochs       = config.get('curriculum_epochs', None)
    early_stopping_patience: Optional[int] = config['early_stopping_patience']

    # ---- Model ----
    model = SiameseNetwork(
        embedding_dim=config['embedding_dim'],
        pretrained=config['pretrained']
    ).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = TripletLoss(margin=config['margin'])

    # ---- Datasets ----
    train_dataset = TripletMoleculeDataset(
        smiles_list=config['train_smiles'],
        image_size=config['image_size'],
        mol_augment=config['mol_augment'],
        default_drawing_style=config['default_drawing_style']
    )
    val_dataset = TripletMoleculeDataset(
        smiles_list=config['val_smiles'],
        shuffle_smiles=False,
        image_size=config['image_size'],
        mol_augment=config['val_mol_augment'],
        default_drawing_style=config['default_drawing_style'],
    )

    # ---- Samplers & DataLoaders ----
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    batch_size = config['batch_size']
    num_workers = config['num_workers']

    def _create_loaders():
        """Create fresh DataLoaders (called on init and after worker crashes)."""
        dl_common: Dict[str, Any] = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'worker_init_fn': _worker_init_fn,
        }
        if num_workers > 0:
            dl_common.update({
                'prefetch_factor': 4,
                'persistent_workers': True,
                'multiprocessing_context': _DL_MP_CONTEXT,
            })
        tl = DataLoader(train_dataset, sampler=train_sampler, timeout=300,
                        **dl_common)
        vl = DataLoader(val_dataset, sampler=val_sampler, timeout=300,
                        **dl_common)
        return tl, vl

    train_loader, val_loader = _create_loaders()

    # ---- Optimizer & Scheduler (warmup → cosine annealing) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=1e-6)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(warmup_steps, total_steps))

    # ---- TensorBoard (rank 0 only) ----
    writer = None
    if is_main:
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        log_dir = _resolve_tb_log_dir(save_path, resume=(start_epoch > 0))
        writer = SummaryWriter(log_dir=log_dir)
        print(f'[Rank 0] TensorBoard logs: {writer.log_dir}')
        print(f'[Rank 0] Batch size per GPU: {batch_size}, GPUs: {world_size}, '
              f'Effective batch size: {batch_size * world_size}')

    # ---- AMP scaler ----
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if is_main:
        print(f'[Rank 0] Total steps: {total_steps}, Warmup steps: {warmup_steps} '
              f'({warmup_ratio*100:.1f}%), AMP={use_amp}')

    # ---- Training-loop state ----
    early_stopping_enabled = early_stopping_patience is not None
    best_val_loss = float('inf')
    vals_no_improve = 0
    global_step = 0

    _curriculum_ramp = _compute_curriculum_ramp(curriculum_epochs, num_epochs)
    curriculum_active = (hard_neg_min < hard_neg_max - 1e-9)
    # For resume: compute prev_curriculum_active at the *previous* epoch so
    # the "curriculum complete" transition detection doesn't re-trigger.
    if start_epoch > 0:
        prev_hard_neg = hard_neg_min + (
            hard_neg_max - hard_neg_min
        ) * min(start_epoch - 1, _curriculum_ramp) / _curriculum_ramp
        prev_curriculum_active = (prev_hard_neg < hard_neg_max - 1e-9)
    else:
        prev_curriculum_active = curriculum_active

    if is_main and curriculum_active:
        print(f'[Rank 0] Curriculum learning: ramping hard_neg_prob over '
              f'{_curriculum_ramp + 1} epoch(s). Best-model tracking and '
              f'early stopping deferred until hard_neg_prob reaches '
              f'{hard_neg_max:.3f}')

    # ---- Resume: restore checkpoint ----
    if start_epoch > 0:
        resume_state_path = os.path.join(save_path, 'training_state.pth')
        resume_model_path = os.path.join(save_path, f'epoch_{start_epoch}.pth')
        if os.path.exists(resume_state_path):
            ckpt = torch.load(resume_state_path, map_location=device,
                              weights_only=False)
            model.module.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if scaler is not None and ckpt.get('scaler_state_dict') is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            global_step = ckpt.get('global_step',
                                   start_epoch * len(train_loader))
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            vals_no_improve = ckpt.get('vals_no_improve', 0)
            if is_main:
                print(f'[Rank 0] Resumed full training state '
                      f'(epoch={start_epoch}, global_step={global_step})')
        elif os.path.exists(resume_model_path):
            model.module.load_state_dict(
                torch.load(resume_model_path, map_location=device,
                           weights_only=True))
            # Fast-forward scheduler to match skipped epochs
            steps_to_skip = start_epoch * len(train_loader)
            for _ in range(steps_to_skip):
                scheduler.step()
            global_step = steps_to_skip
            if is_main:
                print(f'[Rank 0] Resumed model weights from {resume_model_path}, '
                      f'scheduler fast-forwarded to step {global_step}')
        else:
            raise FileNotFoundError(
                f'Cannot resume: neither {resume_state_path} nor '
                f'{resume_model_path} found')

    # ==================== Epoch Loop ====================
    for epoch in range(start_epoch, num_epochs):
        # Curriculum learning: linearly ramp hard-negative probability
        hard_neg_prob = hard_neg_min + (hard_neg_max - hard_neg_min) * min(
            epoch, _curriculum_ramp) / _curriculum_ramp
        train_dataset.hard_negative_prob = hard_neg_prob
        val_dataset.hard_negative_prob = hard_neg_prob
        if is_main:
            print(f'Epoch {epoch+1}: hard_negative_prob={hard_neg_prob:.3f}')

        # --- Train (with automatic DataLoader crash retry) ---
        epoch_loss = 0.0
        epoch_step_count = 0
        for _retry in range(_MAX_WORKER_RETRIES):
            try:
                model.train()
                train_sampler.set_epoch(epoch)
                epoch_loss = 0.0
                epoch_step_count = 0

                train_iter = (
                    tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
                    if is_main else train_loader
                )
                for anchor, positive, negative in train_iter:
                    anchor = anchor.to(device, non_blocking=True)
                    positive = positive.to(device, non_blocking=True)
                    negative = negative.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            a_emb, p_emb, n_emb = model(anchor, positive, negative)
                            loss = criterion(a_emb, p_emb, n_emb)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        a_emb, p_emb, n_emb = model(anchor, positive, negative)
                        loss = criterion(a_emb, p_emb, n_emb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()

                    epoch_loss += loss.item()
                    epoch_step_count += 1
                    if is_main:
                        train_iter.set_postfix({'loss': f'{loss.item():.4f}'})
                        writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                        writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
                    global_step += 1
                break  # epoch completed successfully
            except RuntimeError as e:
                if 'DataLoader worker' in str(e) and _retry < _MAX_WORKER_RETRIES - 1:
                    if is_main:
                        print(f'\n[Rank {rank}] DataLoader worker crash at epoch '
                              f'{epoch+1} (retry {_retry+1}/{_MAX_WORKER_RETRIES}). '
                              f'Recreating DataLoaders...')
                    # Synchronize all ranks before recreating
                    dist.barrier()
                    train_loader, val_loader = _create_loaders()
                    continue
                raise  # non-DataLoader error or retries exhausted

        train_loss = epoch_loss / max(1, epoch_step_count)

        # --- Validate (every epoch) ---
        model.eval()
        val_loss_sum = 0.0
        pos_dist_sum = 0.0
        neg_dist_sum = 0.0
        correct_sum = 0.0
        count = 0.0

        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    a, p, n_ = model(anchor, positive, negative)
                pos_d = F.pairwise_distance(a, p, p=2)
                neg_d = F.pairwise_distance(a, n_, p=2)
                vloss = criterion(a, p, n_)

                bsz = anchor.size(0)
                val_loss_sum += vloss.item() * bsz
                pos_dist_sum += pos_d.sum().item()
                neg_dist_sum += neg_d.sum().item()
                correct_sum += (pos_d < neg_d).sum().item()
                count += bsz

        # Gather metrics across all ranks
        metrics_t = torch.tensor(
            [val_loss_sum, pos_dist_sum, neg_dist_sum, correct_sum, count],
            device=device, dtype=torch.float64
        )
        dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
        t_loss, t_pos, t_neg, t_corr, t_cnt = metrics_t.tolist()

        val_loss = t_loss / max(1, t_cnt)
        avg_pos = t_pos / max(1, t_cnt)
        avg_neg = t_neg / max(1, t_cnt)
        triplet_acc = t_corr / max(1, t_cnt)

        if is_main:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}: train={train_loss:.4f} | val={val_loss:.4f} | '
                  f'pos={avg_pos:.3f} | neg={avg_neg:.3f} | acc={triplet_acc:.3f} | '
                  f'lr={current_lr:.2e} | hard_neg={hard_neg_prob:.3f}')
            writer.add_scalar('Val/loss', val_loss, global_step)
            writer.add_scalar('Val/pos_dist', avg_pos, global_step)
            writer.add_scalar('Val/neg_dist', avg_neg, global_step)
            writer.add_scalar('Val/triplet_acc', triplet_acc, global_step)
            writer.add_scalar('Curriculum/hard_negative_prob', hard_neg_prob, global_step)

            # --- Checkpointing (cache state_dict to avoid redundant copies) ---
            if save_path:
                cached_sd = model.module.state_dict()
                torch.save(cached_sd,
                           os.path.join(save_path, f'epoch_{epoch+1}.pth'))
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': cached_sd,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'vals_no_improve': vals_no_improve,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_path, 'training_state.pth'))

            # --- Curriculum completion detection ---
            curriculum_active = (hard_neg_prob < hard_neg_max - 1e-9)
            if prev_curriculum_active and not curriculum_active:
                best_val_loss = float('inf')
                vals_no_improve = 0
                print(f'  Curriculum complete (hard_neg_prob={hard_neg_prob:.3f}). '
                      f'Resetting best-model tracking and early stopping.')
            prev_curriculum_active = curriculum_active

            # --- Best-model tracking (only after curriculum is stable) ---
            if not curriculum_active:
                if val_loss < best_val_loss:
                    vals_no_improve = 0
                    best_val_loss = val_loss
                    if save_path:
                        torch.save(cached_sd,
                                   os.path.join(save_path, 'best.pth'))
                        print(f'  New best model (val_loss={best_val_loss:.4f}).')
                else:
                    if early_stopping_enabled:
                        vals_no_improve += 1
                        print(f'  No improvement ({vals_no_improve}/{early_stopping_patience}).')
            else:
                # During curriculum ramp, always save latest as best.pth
                # (cross-difficulty comparison is meaningless)
                if save_path:
                    torch.save(cached_sd,
                               os.path.join(save_path, 'best.pth'))
                    print(f'  Curriculum ramping — saved latest as best.pth '
                          f'(val_loss={val_loss:.4f}, not compared).')

        # Broadcast early-stopping decision from rank 0
        # (only fires after curriculum is complete)
        should_stop_flag = (
            early_stopping_enabled
            and not curriculum_active
            and vals_no_improve >= early_stopping_patience
        )
        should_stop = torch.tensor([1 if should_stop_flag else 0],
                                   device=device, dtype=torch.int)
        dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            if is_main:
                print(f'Early stopping at epoch {epoch+1}.')
            break

    # Save final model
    if is_main:
        if save_path:
            torch.save(model.module.state_dict(),
                       os.path.join(save_path, 'final.pth'))
        if writer:
            writer.close()

    # Explicit cleanup to ensure GPU memory is released before process exits
    del model, optimizer, scheduler, criterion
    del train_loader, val_loader, train_dataset, val_dataset
    del train_sampler, val_sampler
    if scaler is not None:
        del scaler
    gc.collect()
    torch.cuda.empty_cache()


class MoleculeDiscriminator:
    """Wrapper class for training and inference."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (384, 384),
        embedding_dim: int = 128,
        margin: float = 1.0,
        device: Optional[torch.device] = None,
        pretrained: bool = True
    ):
        """
        Args:
            image_size: Target image size (height, width)
            embedding_dim: Dimension of embedding space
            margin: Margin for triplet loss
            device: Device for inference (DDP training uses all GPUs automatically)
            pretrained: Whether to use pretrained EfficientNet_V2_S weights
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.pretrained = pretrained

        print(f'Using device: {self.device}')

        self.model = SiameseNetwork(
            embedding_dim=embedding_dim,
            pretrained=pretrained
        ).to(self.device)
        self.criterion = TripletLoss(margin=margin)

        # Inference preprocessing (matches training post_transforms)
        self._pad_to_square = PadToSquare(fill=255)
        self._inference_transforms = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1],
                     interpolation=cv2.INTER_AREA),
            A.ToGray(num_output_channels=3),
            A.Normalize(),  # ImageNet normalization
            ToTensorV2(),
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        smiles_list: List[str],
        num_smiles: Optional[int] = None,
        num_epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 4e-4,
        mol_augment: bool = True,
        save_path: Optional[str] = None,
        val_split: float = 0.01,
        val_mol_augment: bool = False,
        default_drawing_style: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        early_stopping_patience: Optional[int] = 5,
        use_ddp: bool = True,
        warmup_ratio: float = 0.02,
        use_amp: bool = True,
        hard_negative_max_prob: float = 1.0,
        hard_negative_min_prob: float = 0.0,
        curriculum_epochs: Optional[int] = None,
        resume_epoch: int = 0,
    ) -> List[float]:
        """Train the discriminator with train/validation split.

        When ``use_ddp=True`` and multiple CUDA GPUs are available, training is
        launched via ``torch.multiprocessing.spawn`` with
        ``DistributedDataParallel``.  In this mode, epoch losses are logged to
        TensorBoard and the method returns an empty list.  After training the
        best (or final) checkpoint is loaded back into ``self.model``.

        Args:
            smiles_list: Full SMILES strings list.
            num_smiles: Number of SMILES strings to randomly sample from
                *smiles_list* before train/val split.  ``None`` → use all.
                Useful for quick experiments with large datasets.
            num_epochs: Number of training epochs.
            batch_size: Batch size **per GPU**.
            learning_rate: Peak learning rate (after warmup).
            mol_augment: Whether to apply molecule augmentation for training.
            save_path: Directory to save checkpoints (created if needed).
            val_split: Fraction of data reserved for validation.
            val_mol_augment: Whether to apply augmentation on validation.
            default_drawing_style: Whether to use default drawing style.
            num_workers: DataLoader workers (per process).
            seed: Random seed for reproducibility.
            early_stopping_patience: Epochs without val improvement before
                stop.  Set to ``None`` to disable early stopping.
            use_ddp: Use DistributedDataParallel when multiple GPUs available.
            warmup_ratio: Fraction of total steps for linear LR warmup.
            use_amp: Use automatic mixed precision (FP16) to reduce GPU memory.
            hard_negative_max_prob: Maximum probability of generating hard
                negatives.  Linearly ramped from
                ``hard_negative_min_prob`` to this value over
                ``curriculum_epochs``.
            hard_negative_min_prob: Starting probability of generating hard
                negatives.
            curriculum_epochs: Number of epochs over which to ramp hard
                negative probability from ``hard_negative_min_prob`` to
                ``hard_negative_max_prob``.  After this many epochs the
                probability stays at ``hard_negative_max_prob`` for the
                remaining training.  If ``None`` (default), the ramp spans
                half of ``num_epochs``.
            resume_epoch: If > 0, resume training from this epoch.  The
                method first tries to load
                ``<save_path>/training_state.pth`` (full optimizer /
                scheduler / scaler state); if that file is absent it falls
                back to ``<save_path>/epoch_<resume_epoch>.pth`` (model
                weights only, scheduler fast-forwarded).  TensorBoard logs
                are appended to the most recent existing log directory.
                Only supported in DDP mode.

        Returns:
            List of training losses per epoch (empty list in DDP mode —
            see TensorBoard for full logs).
        """
        # ---- Set random seed ----
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ---- Sample SMILES ----
        if num_smiles is not None and num_smiles < len(smiles_list):
            original_n = len(smiles_list)
            rng_sample = np.random.default_rng(seed)
            n_sample = max(2, int(num_smiles))
            sample_idx = rng_sample.choice(original_n, size=n_sample, replace=False)
            smiles_list = [smiles_list[int(i)] for i in sample_idx]
            print(f'Sampled {n_sample}/{original_n} SMILES')

        n = len(smiles_list)
        if n < 2:
            raise ValueError("Need at least 2 SMILES for train/val split.")

        # ---- Train / val split ----
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = max(1, int(n * (1 - val_split)))
        train_idx, val_idx = indices[:split], indices[split:]
        if len(val_idx) == 0:
            val_idx = indices[-1:]
            train_idx = indices[:-1]

        train_smiles = [smiles_list[i] for i in train_idx]
        val_smiles = [smiles_list[i] for i in val_idx]
        print(f'Train: {len(train_smiles)} | Val: {len(val_smiles)}')

        world_size = torch.cuda.device_count()
        if use_ddp and world_size > 1:
            # ---- DDP training ----
            tmp_dir = None
            actual_save = save_path
            if save_path is None:
                tmp_dir = tempfile.mkdtemp(prefix='mol_disc_')
                actual_save = tmp_dir

            config: Dict[str, Any] = {
                'embedding_dim': self.embedding_dim,
                'margin': self.margin,
                'pretrained': self.pretrained,
                'image_size': self.image_size,
                'train_smiles': train_smiles,
                'val_smiles': val_smiles,
                'mol_augment': mol_augment,
                'val_mol_augment': val_mol_augment,
                'default_drawing_style': default_drawing_style,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'early_stopping_patience': early_stopping_patience,
                'save_path': actual_save,
                'master_port': _find_free_port(),
                'warmup_ratio': warmup_ratio,
                'seed': seed,
                'use_amp': use_amp,
                'hard_negative_max_prob': hard_negative_max_prob,
                'hard_negative_min_prob': hard_negative_min_prob,
                'curriculum_epochs': curriculum_epochs,
                'resume_epoch': resume_epoch,
            }

            # Move main-process model to CPU to free GPU 0 memory for worker
            self.model.cpu()
            torch.cuda.empty_cache()

            print(f'Launching DDP training on {world_size} GPUs '
                  f'(port {config["master_port"]})...')
            try:
                mp.spawn(
                    _ddp_train_worker,
                    args=(world_size, config),
                    nprocs=world_size,
                    join=True,
                )
            except KeyboardInterrupt:
                print('\nTraining interrupted by user. Cleaning up...')
            except Exception as e:
                print(f'DDP training failed: {e}')
                raise
            finally:
                # Forcefully clean up GPU memory across all devices
                for i in range(world_size):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                gc.collect()

            # Reload trained weights into the main-process model
            best_path = os.path.join(actual_save, 'best.pth')
            final_path = os.path.join(actual_save, 'final.pth')
            if os.path.exists(best_path):
                self.load_model(best_path)
                print('Loaded best model after DDP training.')
            elif os.path.exists(final_path):
                self.load_model(final_path)
                print('Loaded final model after DDP training.')

            return []  # losses are in TensorBoard
        else:
            # ---- Single-GPU training ----
            return self._train_single(
                train_smiles=train_smiles,
                val_smiles=val_smiles,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mol_augment=mol_augment,
                val_mol_augment=val_mol_augment,
                default_drawing_style=default_drawing_style,
                save_path=save_path,
                num_workers=num_workers,
                early_stopping_patience=early_stopping_patience,
                warmup_ratio=warmup_ratio,
                use_amp=use_amp,
                hard_negative_max_prob=hard_negative_max_prob,
                hard_negative_min_prob=hard_negative_min_prob,
                curriculum_epochs=curriculum_epochs,
            )

    def _train_single(
        self,
        train_smiles: List[str],
        val_smiles: List[str],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        mol_augment: bool,
        val_mol_augment: bool,
        default_drawing_style: bool,
        save_path: Optional[str],
        num_workers: int,
        early_stopping_patience: Optional[int],
        warmup_ratio: float = 0.02,
        use_amp: bool = True,
        hard_negative_max_prob: float = 0.0,
        hard_negative_min_prob: float = 0.0,
        curriculum_epochs: Optional[int] = None,
    ) -> List[float]:
        """Single-GPU / CPU training loop (no DDP).

        Mirrors the DDP training loop but without distributed primitives.
        Uses the same shared helpers (:func:`_make_lr_lambda`,
        :func:`_compute_curriculum_ramp`, :func:`_resolve_tb_log_dir`) for
        consistency.

        Returns:
            List of per-epoch training losses.
        """
        device = self.device

        # ---- Datasets ----
        train_dataset = TripletMoleculeDataset(
            smiles_list=train_smiles,
            image_size=self.image_size,
            mol_augment=mol_augment,
            default_drawing_style=default_drawing_style,
        )
        val_dataset = TripletMoleculeDataset(
            smiles_list=val_smiles,
            shuffle_smiles=False,
            image_size=self.image_size,
            mol_augment=val_mol_augment,
            default_drawing_style=default_drawing_style,
        )

        # ---- DataLoaders ----
        pin_mem = device.type == 'cuda'
        dl_common: Dict[str, Any] = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_mem,
            'worker_init_fn': _worker_init_fn,
        }
        if num_workers > 0:
            dl_common.update({
                'prefetch_factor': 2,
                'persistent_workers': True,
                'multiprocessing_context': _DL_MP_CONTEXT,
            })
        dl_timeout = 300 if num_workers > 0 else 0

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  timeout=dl_timeout, **dl_common)
        val_loader = DataLoader(val_dataset, shuffle=False,
                                timeout=dl_timeout, **dl_common)
        print(f'DataLoader: batch_size={batch_size}, workers={num_workers}, '
              f'device={device}')

        # ---- Optimizer & Scheduler (warmup → cosine annealing) ----
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                      weight_decay=1e-6)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(warmup_steps, total_steps))

        # ---- AMP scaler ----
        use_amp_actual = use_amp and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp_actual else None
        print(f'Total steps: {total_steps}, Warmup steps: {warmup_steps} '
              f'({warmup_ratio*100:.1f}%), AMP={use_amp_actual}')

        # ---- TensorBoard ----
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        log_dir = _resolve_tb_log_dir(save_path, resume=False)
        writer = SummaryWriter(log_dir=log_dir)
        print(f'TensorBoard logs: {writer.log_dir}')

        # ---- Training-loop state ----
        early_stopping_enabled = early_stopping_patience is not None
        best_val_loss = float('inf')
        vals_no_improve = 0
        epoch_losses: List[float] = []
        global_step = 0

        _curriculum_ramp = _compute_curriculum_ramp(curriculum_epochs, num_epochs)
        curriculum_active = (hard_negative_min_prob < hard_negative_max_prob - 1e-9)
        prev_curriculum_active = curriculum_active

        if curriculum_active:
            print(f'Curriculum learning: ramping hard_neg_prob over '
                  f'{_curriculum_ramp + 1} epoch(s). Best-model tracking and '
                  f'early stopping deferred until hard_neg_prob reaches '
                  f'{hard_negative_max_prob:.3f}')

        # ==================== Epoch Loop ====================
        for epoch in range(num_epochs):
            hard_neg_prob = hard_negative_min_prob + (
                hard_negative_max_prob - hard_negative_min_prob
            ) * min(epoch, _curriculum_ramp) / _curriculum_ramp
            train_dataset.hard_negative_prob = hard_neg_prob
            val_dataset.hard_negative_prob = hard_neg_prob
            print(f'Epoch {epoch+1}: hard_negative_prob={hard_neg_prob:.3f}')

            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            train_bar = tqdm(train_loader,
                             desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for anchor, positive, negative in train_bar:
                anchor = anchor.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        a_emb, p_emb, n_emb = self.model(anchor, positive, negative)
                        loss = self.criterion(a_emb, p_emb, n_emb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    a_emb, p_emb, n_emb = self.model(anchor, positive, negative)
                    loss = self.criterion(a_emb, p_emb, n_emb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item()
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
                global_step += 1

            train_loss = epoch_loss / max(1, len(train_loader))
            epoch_losses.append(train_loss)

            # --- Validate ---
            self.model.eval()
            val_loss_sum = 0.0
            pos_dist_sum = 0.0
            neg_dist_sum = 0.0
            correct_sum = 0
            count = 0

            with torch.no_grad():
                for anchor, positive, negative in val_loader:
                    anchor = anchor.to(device, non_blocking=True)
                    positive = positive.to(device, non_blocking=True)
                    negative = negative.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp_actual):
                        a, p, n_ = self.model(anchor, positive, negative)
                    pos_d = F.pairwise_distance(a, p, p=2)
                    neg_d = F.pairwise_distance(a, n_, p=2)
                    vloss = self.criterion(a, p, n_)

                    bsz = anchor.size(0)
                    val_loss_sum += vloss.item() * bsz
                    pos_dist_sum += pos_d.sum().item()
                    neg_dist_sum += neg_d.sum().item()
                    correct_sum += (pos_d < neg_d).sum().item()
                    count += bsz

            val_loss = val_loss_sum / max(1, count)
            avg_pos = pos_dist_sum / max(1, count)
            avg_neg = neg_dist_sum / max(1, count)
            triplet_acc = correct_sum / max(1, count)

            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}: train={train_loss:.4f} | val={val_loss:.4f} | '
                  f'pos={avg_pos:.3f} | neg={avg_neg:.3f} | acc={triplet_acc:.3f} | '
                  f'lr={current_lr:.2e} | hard_neg={hard_neg_prob:.3f}')
            writer.add_scalar('Val/loss', val_loss, global_step)
            writer.add_scalar('Val/pos_dist', avg_pos, global_step)
            writer.add_scalar('Val/neg_dist', avg_neg, global_step)
            writer.add_scalar('Val/triplet_acc', triplet_acc, global_step)
            writer.add_scalar('Curriculum/hard_negative_prob', hard_neg_prob, global_step)

            # --- Checkpointing (cache state_dict) ---
            if save_path:
                cached_sd = self.model.state_dict()
                torch.save(cached_sd,
                           os.path.join(save_path, f'epoch_{epoch+1}.pth'))

            # --- Curriculum completion detection ---
            curriculum_active = (hard_neg_prob < hard_negative_max_prob - 1e-9)
            if prev_curriculum_active and not curriculum_active:
                best_val_loss = float('inf')
                vals_no_improve = 0
                print(f'  Curriculum complete (hard_neg_prob={hard_neg_prob:.3f}). '
                      f'Resetting best-model tracking and early stopping.')
            prev_curriculum_active = curriculum_active

            # --- Best-model tracking (only after curriculum is stable) ---
            if not curriculum_active:
                if val_loss < best_val_loss:
                    vals_no_improve = 0
                    best_val_loss = val_loss
                    if save_path:
                        torch.save(cached_sd,
                                   os.path.join(save_path, 'best.pth'))
                        print(f'  New best model (val_loss={best_val_loss:.4f}).')
                else:
                    if early_stopping_enabled:
                        vals_no_improve += 1
                        print(f'  No improvement ({vals_no_improve}/{early_stopping_patience}).')
            else:
                # During curriculum ramp, always save latest as best.pth
                if save_path:
                    torch.save(cached_sd,
                               os.path.join(save_path, 'best.pth'))
                    print(f'  Curriculum ramping — saved latest as best.pth '
                          f'(val_loss={val_loss:.4f}, not compared).')

            if (early_stopping_enabled
                    and not curriculum_active
                    and vals_no_improve >= early_stopping_patience):
                print(f'Early stopping at epoch {epoch+1}.')
                break

        # Save final model
        if save_path:
            torch.save(self.model.state_dict(),
                       os.path.join(save_path, 'final.pth'))
        writer.close()
        return epoch_losses

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess a single image for inference (matches training pipeline)."""
        img, _ = self._pad_to_square(img)
        result = self._inference_transforms(image=img)
        return result['image']

    def predict(
        self, 
        img1, 
        img2,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Predict whether two images represent the same compound.
        
        Args:
            img1: First image (RGB numpy array or preprocessed torch.Tensor)
            img2: Second image (RGB numpy array or preprocessed torch.Tensor)
            threshold: Distance threshold for same/different decision
            
        Returns:
            (is_same, distance) where is_same is True if predicted to be same compound
        """
        self.model.eval()
        
        if isinstance(img1, torch.Tensor):
            img1_tensor = img1.unsqueeze(0).to(self.device) if img1.dim() == 3 else img1.to(self.device)
        else:
            img1_tensor = self._preprocess_image(img1).unsqueeze(0).to(self.device)
        
        if isinstance(img2, torch.Tensor):
            img2_tensor = img2.unsqueeze(0).to(self.device) if img2.dim() == 3 else img2.to(self.device)
        else:
            img2_tensor = self._preprocess_image(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb1 = self.model.forward_once(img1_tensor)
            emb2 = self.model.forward_once(img2_tensor)
        
        distance = F.pairwise_distance(emb1, emb2, p=2).item()
        is_same = distance < threshold
        
        return is_same, distance

    def get_embeddings(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get embeddings for a list of images using all available GPUs.

        Args:
            images: List of RGB numpy arrays
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (N, embedding_dim)
        """
        self.model.eval()

        # Wrap for multi-GPU inference via DataParallel
        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            embed_model = nn.DataParallel(
                _EmbeddingWrapper(self.model)
            ).to(self.device)
        else:
            embed_model = _EmbeddingWrapper(self.model).to(self.device)
        embed_model.eval()

        tensors = torch.stack([self._preprocess_image(img) for img in images])

        all_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = tensors[i:i + batch_size].to(self.device)
                emb = embed_model(batch)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def predict_batch(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> List[Tuple[bool, float]]:
        """
        Batch prediction for multiple image pairs using all available GPUs.

        Internally uses ``nn.DataParallel`` to distribute the embedding
        computation across GPUs.

        Args:
            image_pairs: List of (img1, img2) tuples (RGB numpy arrays)
            threshold: Distance threshold for same/different decision
            batch_size: Processing batch size

        Returns:
            List of (is_same, distance) tuples
        """
        img1_list = [pair[0] for pair in image_pairs]
        img2_list = [pair[1] for pair in image_pairs]

        emb1 = self.get_embeddings(img1_list, batch_size=batch_size)
        emb2 = self.get_embeddings(img2_list, batch_size=batch_size)

        emb1_t = torch.from_numpy(emb1)
        emb2_t = torch.from_numpy(emb2)
        distances = F.pairwise_distance(emb1_t, emb2_t, p=2).numpy()

        return [(float(d) < threshold, float(d)) for d in distances]

    def load_model(self, path: str):
        """Load trained model weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
