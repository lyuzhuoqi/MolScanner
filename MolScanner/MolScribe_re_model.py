"""
MolScanner Model — DistributedDataParallel (DDP) Training & Inference
=====================================================================

End-to-end molecule-image recognition model:
  Image  →  SequenceDecoder (Transformer, autoregressive)  →  SMILES + atom coordinates
         →  BondPredictor   (MLP, pairwise)                →  bond matrix

Sequence format: **chartok_coords**
  [SOS, <SMILES chars>, X_BIN, Y_BIN, <SMILES chars>, X_BIN, Y_BIN, …, EOS]
  Each atom's characters are followed by its binned (x, y) image coordinates.

Three SMILES output modes for evaluation / inference:
  • decoder     – raw SMILES from the decoded token sequence
  • graph       – SMILES reconstructed from predicted atoms + bonds
  • postprocess – MolScribe-style: replace functional groups, restore stereo

Launch training with ``torchrun``:
  torchrun --nproc_per_node=<N_GPUS> train_MolScanner_ddp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd

from typing import Tuple, List, Optional, Dict
import numpy as np
import random
import string
import os
from functools import partial
from func_timeout import FunctionTimedOut
import math
import warnings
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.tensorboard.writer import SummaryWriter

# ======================== Performance Optimizations ========================
# Enable cuDNN benchmark: safe because image_size is fixed (384x384)
# Gives ~10-20% speedup on convolutions by auto-tuning kernel selection
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Use TensorFloat-32 for faster matmul operations on Ampere+ GPUs (no accuracy loss for training)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
# Disable cv2 threading globally to avoid conflicts with DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from drawing_engine import generate_image_from_smiles, _blank_image
from chemistry import (
    _convert_graph_to_smiles, _verify_chirality, canonicalize_smiles,
    _replace_functional_group, _expand_functional_group,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from SmilesPE.pretokenizer import atomwise_tokenizer

# Three SMILES prediction modes (used in validation / evaluation / inference)
SMILES_MODE_DECODER = 'decoder'        # SMILES directly from decoder sequence
SMILES_MODE_GRAPH   = 'graph'          # SMILES reconstructed from predicted atoms + bonds
SMILES_MODE_POSTPROCESS = 'postprocess' # decoder SMILES + chirality correction via predicted coords/edges

# ======================== DDP Helpers ========================

def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group with tuned timeouts.

    Timeout budget (defence-in-depth):
      Layer 1: func_set_timeout(5s) in drawing_engine  – catches most hangs
      Layer 2: DataLoader timeout=60s                  – kills hung C-extension workers
      Layer 3: NCCL collective timeout=5 min           – fast crash if recovery fails
      Layer 4: NCCL heartbeat env var=1800s            – prevents watchdog kills during recovery
    """
    # Prevent NCCL watchdog from killing processes while DataLoader is recovering
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Best-effort destroy of distributed process group.

    In rare cases NCCL teardown can raise (e.g., transient CUDA OOM during
    communicator shutdown). Since training/evaluation has already finished at
    this point, avoid turning cleanup-time backend issues into hard failures.
    """
    if not dist.is_initialized():
        return

    try:
        dist.destroy_process_group()
    except Exception as e:
        warnings.warn(f"cleanup_ddp() ignored backend shutdown error: {e}")


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


# ======================== Vocabulary & Tokenizer ========================

class MolScannerVocab:
    """Character-level vocabulary for chartok_coords sequences.

    Token layout: [special] + [SMILES chars] + [X_BIN_0 … X_BIN_{n-1}] + [Y_BIN_0 … Y_BIN_{n-1}]
    Coordinates are quantized into *n_bins* bins per axis.
    """
    
    def __init__(self, n_bins: int = 64):
        """
        Args:
            n_bins: Number of bins for coordinate quantization (0 to n_bins-1)
        """
        self.n_bins = n_bins
        
        # Special tokens (Removed wrapper tokens)
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        
        special_tokens = [self.PAD, self.SOS, self.EOS, self.UNK]
        
        # SMILES character set
        uppercase = string.ascii_uppercase  # A-Z
        lowercase = string.ascii_lowercase  # a-z
        digits = string.digits  # 0-9
        symbols = ['[', ']', '(', ')', '=', '#', '@', '+', '-', '/', '\\', '.', '%']
        
        smiles_chars = list(uppercase) + list(lowercase) + list(digits) + symbols
        
        # Separate X and Y coordinate bins (0 to n_bins-1)
        x_coord_bins = [f'<X_BIN_{i}>' for i in range(n_bins)]
        y_coord_bins = [f'<Y_BIN_{i}>' for i in range(n_bins)]
        
        # Build vocab: [special] + [smiles_chars] + [X_bins] + [Y_bins]
        self.tokens = special_tokens + smiles_chars + x_coord_bins + y_coord_bins
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        self.pad_idx = self.token2idx[self.PAD]
        self.sos_idx = self.token2idx[self.SOS]
        self.eos_idx = self.token2idx[self.EOS]
        self.unk_idx = self.token2idx[self.UNK]

        # Pre-compute bin token ID ranges for fast checking
        self.n_smiles_chars = len(smiles_chars)
        self.n_special = len(special_tokens)
        
        # X bins range
        self.x_bin_start_idx = self.n_special + self.n_smiles_chars
        self.x_bin_end_idx = self.x_bin_start_idx + n_bins - 1
        
        # Y bins range
        self.y_bin_start_idx = self.x_bin_end_idx + 1
        self.y_bin_end_idx = self.y_bin_start_idx + n_bins - 1
        
        # Combined range for backward compatibility
        self.bin_start_idx = self.x_bin_start_idx
        self.bin_end_idx = self.y_bin_end_idx
        
    def __len__(self):
        return len(self.tokens)

    def is_coord_token(self, idx: int) -> bool:
        """Check if a token index represents any coordinate bin (X or Y)."""
        return self.bin_start_idx <= idx <= self.bin_end_idx
    
    def is_x_coord_token(self, idx: int) -> bool:
        """Check if a token index represents an X coordinate bin."""
        return self.x_bin_start_idx <= idx <= self.x_bin_end_idx
    
    def is_y_coord_token(self, idx: int) -> bool:
        """Check if a token index represents a Y coordinate bin."""
        return self.y_bin_start_idx <= idx <= self.y_bin_end_idx
    
    def is_symbol(self, idx: int) -> bool:
        """Check if token index represents a SMILES symbol (not special, not coordinate)."""
        return self.n_special <= idx < self.x_bin_start_idx

    @staticmethod
    def is_atom_token(token: str) -> bool:
        """Check if a SMILES token (from atomwise_tokenizer) represents an atom."""
        return token.isalpha() or token.startswith("[") or token == '*'

    def smiles_to_sequence(self, smiles: str, coords: Optional[List] = None) -> Tuple[List[int], List[int]]:
        """
        Tokenize SMILES with interleaved coordinates (chartok_coords style).
        
        Format: [SOS, ..., atom_chars, X_BIN, Y_BIN, bond_char, atom_chars, X_BIN, Y_BIN, ..., EOS]
        Coordinates are inserted after each atom token's characters.
        
        Args:
            smiles: SMILES string
            coords: List of [x_bin, y_bin] for each atom, in SMILES atom order.
                    Values should already be in range [0, n_bins-1].
        Returns:
            labels: List of token indices
            indices: List of positions of Y_BIN tokens (one per atom, for bond predictor)
        """
        tokens = atomwise_tokenizer(smiles)
        labels = [self.sos_idx]
        indices = []
        atom_idx = -1

        for token in tokens:
            # Tokenize each character of the SMILES token
            for c in token:
                if c in self.token2idx:
                    labels.append(self.token2idx[c])
                else:
                    labels.append(self.unk_idx)

            # If this token is an atom, append X_BIN and Y_BIN
            if self.is_atom_token(token):
                atom_idx += 1
                if coords is not None and atom_idx < len(coords):
                    x_bin = max(0, min(self.n_bins - 1, int(coords[atom_idx][0])))
                    y_bin = max(0, min(self.n_bins - 1, int(coords[atom_idx][1])))
                    labels.append(self.token2idx[f'<X_BIN_{x_bin}>'])
                    labels.append(self.token2idx[f'<Y_BIN_{y_bin}>'])
                else:
                    # Fallback: use bin 0 (should not happen in normal training)
                    labels.append(self.token2idx['<X_BIN_0>'])
                    labels.append(self.token2idx['<Y_BIN_0>'])
                indices.append(len(labels) - 1)  # Position of Y_BIN

        labels.append(self.eos_idx)
        return labels, indices

    def sequence_to_smiles(self, sequence: List[int]) -> Dict:
        """
        Detokenize a chartok_coords sequence back to SMILES + coords + symbols.
        
        Returns:
            dict with keys:
                'smiles': reconstructed SMILES string
                'symbols': list of atom symbol strings
                'coords': list of [x_bin, y_bin] per atom
                'indices': list of Y_BIN positions in the sequence (for bond predictor)
                'success': bool
        """
        smiles = ''
        coords, symbols, indices = [], [], []
        i = 0

        if len(sequence) > 0 and sequence[0] == self.sos_idx:
            i = 1

        while i < len(sequence):
            label = sequence[i]
            if label == self.eos_idx or label == self.pad_idx:
                break
            # Skip coordinate tokens (they are consumed when following an atom)
            if self.is_x_coord_token(label) or self.is_y_coord_token(label):
                i += 1
                continue
            # Skip special tokens
            if label in (self.pad_idx, self.sos_idx, self.unk_idx):
                i += 1
                continue

            token_str = self.idx2token.get(label, '')

            # --- Bracket atom: [...]  ---
            if token_str == '[':
                j = i + 1
                while j < len(sequence):
                    jt = self.idx2token.get(sequence[j], '')
                    if not self.is_symbol(sequence[j]):
                        break
                    if jt == ']':
                        j += 1
                        break
                    j += 1
                atom_token = ''.join(self.idx2token.get(sequence[k], '') for k in range(i, j))
                smiles += atom_token
                # Read following coords
                if (j + 1 < len(sequence)
                        and self.is_x_coord_token(sequence[j])
                        and self.is_y_coord_token(sequence[j + 1])):
                    x_val = int(self.idx2token[sequence[j]][7:-1])
                    y_val = int(self.idx2token[sequence[j + 1]][7:-1])
                    coords.append([x_val, y_val])
                    symbols.append(atom_token)
                    indices.append(j + 1)
                    i = j + 2
                else:
                    i = j

            # --- Regular atom (uppercase letter, or lowercase aromatic atom) ---
            elif token_str.isalpha():
                j = i + 1
                # Check for two-letter atoms: Cl, Br (uppercase), se, te (aromatic)
                if (j < len(sequence) and self.is_symbol(sequence[j])):
                    next_ch = self.idx2token.get(sequence[j], '')
                    if ((token_str == 'C' and next_ch == 'l')
                            or (token_str == 'B' and next_ch == 'r')
                            or (token_str == 's' and next_ch == 'e')
                            or (token_str == 't' and next_ch == 'e')):
                        j = i + 2
                atom_token = ''.join(self.idx2token.get(sequence[k], '') for k in range(i, j))
                smiles += atom_token
                # Read following coords
                if (j + 1 < len(sequence)
                        and self.is_x_coord_token(sequence[j])
                        and self.is_y_coord_token(sequence[j + 1])):
                    x_val = int(self.idx2token[sequence[j]][7:-1])
                    y_val = int(self.idx2token[sequence[j + 1]][7:-1])
                    coords.append([x_val, y_val])
                    symbols.append(atom_token)
                    indices.append(j + 1)
                    i = j + 2
                else:
                    i = j

            # --- Non-atom symbol (bond chars: =, #, (, ), digits, etc.) ---
            else:
                smiles += token_str
                i += 1

        success = len(symbols) > 0
        return {
            'smiles': smiles,
            'symbols': symbols,
            'coords': coords,
            'indices': indices,
            'success': success
        }

    def get_output_mask(self, token_idx: int) -> List[bool]:
        """
        Get output constraint mask for the next token given current token.
        Returns a list of bools where True = disallowed.
        
        Rules:
            After X_BIN  → only Y_BIN is allowed
            After Y_BIN  → no coordinate tokens allowed (symbols or EOS)
            Otherwise    → no Y_BIN allowed (must go through X first)
        """
        mask = [False] * len(self)
        if self.is_x_coord_token(token_idx):
            # After X_BIN → only Y_BIN
            for i in range(len(self)):
                if not self.is_y_coord_token(i):
                    mask[i] = True
        elif self.is_y_coord_token(token_idx):
            # After Y_BIN → no coords, no PAD/SOS
            for i in range(self.x_bin_start_idx, self.y_bin_end_idx + 1):
                mask[i] = True
            mask[self.pad_idx] = True
            mask[self.sos_idx] = True
        else:
            # After symbol/SOS → no Y_BIN (must produce X first if starting coords)
            for i in range(self.y_bin_start_idx, self.y_bin_end_idx + 1):
                mask[i] = True
            mask[self.pad_idx] = True
            mask[self.sos_idx] = True
        return mask

# ======================== CropWhite for Training ========================

class CropWhiteTrain(A.DualTransform):
    """Crop white borders and re-pad with a small margin.
    
    Adapted from MolScribe's CropWhite for albumentations 2.0 API.
    Supports keypoint tracking (DualTransform).
    
    After geometric augmentations (e.g., rotation with fit_output=True),
    the image may have large white corners. This transform trims them so
    the molecule fills the canvas, then adds a small uniform padding.
    """
    
    def __init__(self, value=(255, 255, 255), pad=5, p=1.0):
        super().__init__(p=p)
        self.value = value
        self.pad = pad

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        height, width = img.shape[:2]
        
        # Find non-white bounding box
        if img.ndim == 3:
            x = (img != np.array(self.value)).any(axis=2)
        else:
            x = (img != self.value[0])
        
        if not x.any():
            return {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}
        
        row_sum = x.any(axis=1)
        col_sum = x.any(axis=0)
        
        top = int(row_sum.argmax())
        bottom = height - int(row_sum[::-1].argmax())
        left = int(col_sum.argmax())
        right = width - int(col_sum[::-1].argmax())
        
        return {
            "crop_top": top,
            "crop_bottom": height - bottom,
            "crop_left": left,
            "crop_right": width - right,
        }

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width = img.shape[:2]
        cropped = img[crop_top:height - max(crop_bottom, 0) or height,
                      crop_left:width - max(crop_right, 0) or width]
        # Re-pad with uniform small margin
        if self.pad > 0:
            if cropped.ndim == 3:
                fill_val = self.value
            else:
                fill_val = (self.value[0],)
            cropped = cv2.copyMakeBorder(
                cropped, self.pad, self.pad, self.pad, self.pad,
                cv2.BORDER_CONSTANT, value=fill_val
            )
        return cropped

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
        **params,
    ) -> np.ndarray:
        result = keypoints.copy()
        result[:, 0] = result[:, 0] - crop_left + self.pad
        result[:, 1] = result[:, 1] - crop_top + self.pad
        return result

    def get_transform_init_args_names(self):
        return ('value', 'pad')

# ======================== Dataset ========================

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

class MoleculeDataset(Dataset):
    """
    Dataset that renders SMILES into molecule images and produces
    chartok_coords token sequences + bond matrices for training.
    """
    
    def __init__(
        self, 
        vocab,
        smiles_list: List[str],
        shuffle_smiles: bool = True,
        image_size: Tuple[int, int] = (384, 384),
        mol_augment: bool = True,
        img_augment: bool = True,
        geo_augment: bool = True,
        default_drawing_style: bool = False,
    ):
        self.vocab = vocab
        self.smiles_list = smiles_list
        
        if shuffle_smiles:
            if is_main_process():
                print('Shuffling SMILES list for dataset initialization...')
            rng = np.random.default_rng(42)
            shuffled_indices = rng.permutation(len(self.smiles_list))
            self.smiles_list = [self.smiles_list[i] for i in shuffled_indices]
            if is_main_process():
                print('SMILES list shuffled.')

        self.image_size = image_size
        self.mol_augment = mol_augment
        self.img_augment = img_augment
        self.geo_augment = geo_augment
        self.default_drawing_style = default_drawing_style
        
        # Separate pad to square operation (handles keypoints manually)
        self.pad_to_square = PadToSquare(fill=255)

        # 1. Geometric augmentations (Albumentations handles keypoints)
        self.geo_transforms_list = []
        if self.geo_augment:
            self.geo_transforms_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=0.5),
                CropWhiteTrain(pad=5, p=1.0),  # Trim white corners from rotation before further augmentation
                A.RandomCropFromBorders(crop_left=0.01, crop_right=0.01, crop_top=0.01, crop_bottom=0.01, p=0.5),
                A.CropAndPad(percent=(0.0, 0.4), sample_independently=True, keep_size=False, fill=255, fill_mask=255, p=0.2),
            ]
        self.geo_transforms = A.Compose(self.geo_transforms_list, 
                                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
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
            AdaptiveResize(height=self.image_size[0], width=self.image_size[1]),  # INTER_LINEAR↑ / INTER_AREA↓
            A.ToGray(num_output_channels=3),  # Keep 3 channels for pretrained backbone
            A.Normalize(),  # ImageNet normalization
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        anchor_smiles = self.smiles_list[idx]

        # Initialize defaults to handle exceptions safely
        img = _blank_image()
        success = False
        graph = None
        output_smiles = None

        try:
            img, output_smiles, graph, success, _, _ = generate_image_from_smiles(
                anchor_smiles, n_bins=None, 
                mol_augment=self.mol_augment, 
                default_drawing_style=self.default_drawing_style,
                debug=False
            )
        except (FunctionTimedOut, Exception):
            # Catch FunctionTimedOut and other rendering errors
            success = False
            graph = None

        if not success or graph is None or output_smiles is None:
            # Fallback: just return empty with a dummy image
            dummy_img = _blank_image()
            dummy_img, _ = self.pad_to_square(dummy_img)
            img_tensor = self.post_transforms(image=dummy_img, keypoints=[])['image']
            return {'img': img_tensor, 
                    'tok_id_seq': [], 
                    'atom_indices': [],
                    'bond_mat': np.zeros((1, 1), dtype=int), 
                    'success': False}

        coords = graph['coords']  # list of [x, y] in pixel space
        bond_mat = graph['edges']
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Apply geo-transforms (rotation, etc.) with keypoints
        geo_result = self.geo_transforms(image=img, keypoints=coords)
        img = geo_result['image']
        coords = geo_result['keypoints']
        # Step 2: Pad to square (manual, handles keypoints)
        img, coords = self.pad_to_square(img, coords)
        # Step 3: Apply image transforms (blur, noise, etc.)
        img_result = self.img_transforms(image=img)
        img = img_result['image']
        # Step 4: Apply post-transforms (grayscale, normalize, to tensor)
        post_result = self.post_transforms(image=img, keypoints=coords)
        img_tensor = post_result['image']
        coords = post_result['keypoints']
        
        # Bin coordinates to [0, n_bins-1]
        H, W = self.image_size
        n_bins = self.vocab.n_bins
        binned_coords = []
        for coord in coords:
            x = min(max(0, coord[0]), W - 1)
            y = min(max(0, coord[1]), H - 1)
            x_bin = int((x / W) * n_bins) 
            y_bin = int((y / H) * n_bins)
            x_bin = max(0, min(n_bins - 1, x_bin))
            y_bin = max(0, min(n_bins - 1, y_bin))
            binned_coords.append([x_bin, y_bin])

        # Tokenize SMILES with interleaved coordinates (chartok_coords)
        try:
            tok_id_seq, atom_indices = self.vocab.smiles_to_sequence(output_smiles, binned_coords)
        except Exception:
            # Fallback on tokenization failure
            dummy_img = _blank_image()
            dummy_img, _ = self.pad_to_square(dummy_img)
            img_tensor = self.post_transforms(image=dummy_img, keypoints=[])['image']
            return {'img': img_tensor, 
                    'tok_id_seq': [], 
                    'atom_indices': [],
                    'bond_mat': np.zeros((1, 1), dtype=int), 
                    'success': False}

        return {
            'img': img_tensor,
            'tok_id_seq': tok_id_seq,
            'atom_indices': atom_indices,
            'bond_mat': bond_mat,
            'success': True
        }

# ======================== USPTO Mol-File Dataset ========================

def normalize_nodes(nodes: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """Normalize node coordinates to [0, 1] based on bounding box.

    Args:
        nodes: (N, 2) array of coordinates.
        flip_y: If True, flip Y axis (image convention: y=0 at top).
    """
    x, y = nodes[:, 0].copy(), nodes[:, 1].copy()
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


class USPTOMolDataset(Dataset):
    """Dataset that loads patent images + coordinates/edges from a CSV file.

    Each row in the CSV has columns:
        file_path   – path to the molecule image (absolute, or relative to *data_dir*)
        SMILES      – ground truth SMILES string
        node_coords – stringified list of [x, y] coordinates (in mol-file space)
        edges       – (optional) stringified list of [u, v, bond_type] triples

    Coordinates from mol files are in arbitrary space. They are normalised to
    [0,1] via ``normalize_nodes``, then scaled to image pixel space before
    augmentation.  After augmentation they are re-normalised to [0,1] and
    quantised into vocab bins—exactly mirroring the dynamic-generation path
    in ``MoleculeDataset``.
    """

    def __init__(
        self,
        vocab: MolScannerVocab,
        csv_path: str,
        data_dir: str,
        image_size: Tuple[int, int] = (384, 384),
        img_augment: bool = True,
        geo_augment: bool = True,
    ):
        self.vocab = vocab
        self.image_size = image_size
        self.df = pd.read_csv(csv_path)

        # Filter out rows whose SMILES contain characters outside the vocab
        # (e.g. "*")
        vocab_chars = set(self.vocab.token2idx.keys())
        def _has_unk(smiles: str) -> bool:
            return any(c not in vocab_chars for c in str(smiles))
        def _has_annotation(smiles: str) -> bool:
            return any(segment in ['[(]', '[)]']for segment in smiles.split('.'))
        n_before = len(self.df)
        self.df = self.df[~self.df['SMILES'].apply(_has_unk)&~self.df['SMILES'].apply(_has_annotation)].reset_index(drop=True)
        n_filtered = n_before - len(self.df)
        if n_filtered > 0:
            print(f"[USPTOMolDataset] Filtered {n_filtered}/{n_before} rows "
                  f"with non-vocab SMILES chars ({100*n_filtered/n_before:.1f}%)")

        # Resolve image paths
        if not os.path.isabs(self.df['file_path'].iloc[0]):
            self.df['file_path'] = self.df['file_path'].apply(
                lambda p: os.path.join(data_dir, p)
            )

        self.pad_to_square = PadToSquare(fill=255)

        # Geometric augmentations (with keypoint tracking)
        geo_list = []
        if geo_augment:
            geo_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=0.5),
                CropWhiteTrain(pad=5, p=1.0),
                A.RandomCropFromBorders(
                    crop_left=0.01, crop_right=0.01,
                    crop_top=0.01, crop_bottom=0.01, p=0.5,
                ),
                A.CropAndPad(
                    percent=(0.0, 0.4), sample_independently=True,
                    keep_size=False, fill=255, fill_mask=255, p=0.2,
                ),
            ]
        self.geo_transforms = A.Compose(
            geo_list,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

        # Image-level noise augmentation
        img_list = []
        if img_augment:
            img_list += [
                A.Downscale(scale_range=(0.5, 0.8),
                            interpolation_pair={'upscale': 3, 'downscale': 3}),
                A.Blur(),
                A.GaussNoise(),
                A.SaltAndPepper(),
            ]
        self.img_transforms = A.Compose(img_list)

        # Final resize + normalise
        self.post_transforms = A.Compose([
            AdaptiveResize(height=image_size[0], width=image_size[1]),
            A.ToGray(num_output_channels=3),
            A.Normalize(),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df.iloc[idx]

        # --- Load image ---
        img = cv2.imread(row['file_path'])
        if img is None:
            return self._dummy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        smiles = row['SMILES']
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return self._dummy()

        # --- Parse node coordinates ---
        try:
            raw_coords = np.array(eval(row['node_coords']), dtype=np.float64)
        except Exception:
            return self._dummy()
        if raw_coords.ndim != 2 or raw_coords.shape[1] != 2 or len(raw_coords) == 0:
            return self._dummy()

        # Normalise to [0,1] then scale to image pixel space
        coords_norm = normalize_nodes(raw_coords)  # [0,1], y-flipped
        pixel_coords = coords_norm.copy()
        pixel_coords[:, 0] *= w
        pixel_coords[:, 1] *= h
        pixel_coords = pixel_coords.tolist()

        # --- Parse edge list → N×N bond matrix ---
        n_atoms = len(raw_coords)
        bond_mat = np.zeros((n_atoms, n_atoms), dtype=int)
        if 'edges' in self.df.columns:
            try:
                edge_list = eval(row['edges'])
                for u, v, t in edge_list:
                    if u < n_atoms and v < n_atoms:
                        if t <= 4:
                            bond_mat[u, v] = t
                            bond_mat[v, u] = t
                        else:
                            bond_mat[u, v] = t
                            bond_mat[v, u] = 11 - t
            except Exception:
                pass  # fall back to zero matrix

        # --- Augment ---
        geo_result = self.geo_transforms(image=img, keypoints=pixel_coords)
        img = geo_result['image']
        coords_px = geo_result['keypoints']

        img, coords_px = self.pad_to_square(img, coords_px)
        img = self.img_transforms(image=img)['image']

        post_result = self.post_transforms(image=img, keypoints=coords_px)
        img_tensor = post_result['image']
        coords_px = post_result['keypoints']

        # --- Quantise coordinates to bins ---
        H, W = self.image_size
        n_bins = self.vocab.n_bins
        binned_coords = []
        for coord in coords_px:
            x = min(max(0, coord[0]), W - 1)
            y = min(max(0, coord[1]), H - 1)
            x_bin = max(0, min(n_bins - 1, int((x / W) * n_bins)))
            y_bin = max(0, min(n_bins - 1, int((y / H) * n_bins)))
            binned_coords.append([x_bin, y_bin])

        # --- Tokenise ---
        try:
            tok_id_seq, atom_indices = self.vocab.smiles_to_sequence(smiles, binned_coords)
        except Exception:
            return self._dummy()

        return {
            'img': img_tensor,
            'smiles': smiles,
            'tok_id_seq': tok_id_seq,
            'atom_indices': atom_indices,
            'bond_mat': bond_mat,
            'success': True,
        }

    # ------------------------------------------------------------------

    def _dummy(self) -> Dict:
        dummy = _blank_image()
        dummy, _ = self.pad_to_square(dummy)
        img_tensor = self.post_transforms(image=dummy, keypoints=[])['image']
        return {
            'img': img_tensor,
            'smiles': '',
            'tok_id_seq': [],
            'atom_indices': [],
            'bond_mat': np.zeros((1, 1), dtype=int),
            'success': False,
        }


# ======================== Collate Function ========================

def collate_fn(batch_list: List[Dict], vocab: MolScannerVocab, max_atoms_limit: int = 100, max_seq_len: int = 600) -> Optional[Dict]:
    """
    Collate function for chartok_coords format.
    Sequences are pre-tokenized in the dataset; this function pads and batches them.
    """
    # 1. Stack images
    batch_list = [item for item in batch_list if item is not None]
    if len(batch_list) == 0: return None
    
    # ===== Filter out failed samples and molecules with too many atoms / too long sequences =====
    filtered_batch = []
    for item in batch_list:
        # 1. Skip samples that failed rendering / tokenization
        if not item.get('success', True) or len(item['tok_id_seq']) == 0:
            continue
        # 2. Skip molecules exceeding the atom-count limit
        n_atoms = len(item['atom_indices'])
        if n_atoms <= 0 or n_atoms > max_atoms_limit:
            continue
        # 3. Skip molecules with excessively long token sequences (prevents OOM in attention)
        if len(item['tok_id_seq']) > max_seq_len:
            continue
        filtered_batch.append(item)
    
    if len(filtered_batch) == 0:
        return None
    
    batch_list = filtered_batch
    # ========================================
    
    batch_size = len(batch_list)
    images = torch.stack([item['img'] for item in batch_list])
    
    # 2. Pad pre-tokenized sequences
    tokenized_seqs = [item['tok_id_seq'] for item in batch_list]
    
    max_len = max(len(seq) for seq in tokenized_seqs)
    tgt_tokens = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long)
    tgt_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    
    for i, tokens in enumerate(tokenized_seqs):
        seq_len = len(tokens)
        tgt_tokens[i, :seq_len] = torch.tensor(tokens, dtype=torch.long)
        tgt_padding_mask[i, :seq_len] = False
    
    # 3. Pad atom indices (pre-computed in dataset)
    atom_indices_list = [item['atom_indices'] for item in batch_list]
    max_atoms = max(len(indices) for indices in atom_indices_list)
    max_atoms = max(max_atoms, 1)
    
    atom_indices = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    
    for i, indices in enumerate(atom_indices_list):
        n_atoms = len(indices)
        if n_atoms > 0:
            valid_indices = [idx for idx in indices if idx < max_len]
            n_valid = len(valid_indices)
            if n_valid > 0:
                atom_indices[i, :n_valid] = torch.tensor(valid_indices, dtype=torch.long)
                atom_mask[i, :n_valid] = True
    
    # 4. Collect bond matrices (also truncate to max_atoms_limit)
    bond_matrices_list = []
    for item in batch_list:
        bond_mat = item['bond_mat']
        n = bond_mat.shape[0]
        if n > max_atoms_limit:
            bond_mat = bond_mat[:max_atoms_limit, :max_atoms_limit]
        bond_matrices_list.append(bond_mat)
    
    return {
        'images': images,
        'tgt_tokens': tgt_tokens,
        'tgt_padding_mask': tgt_padding_mask,
        'atom_indices': atom_indices,
        'atom_mask': atom_mask,
        'max_atoms': max_atoms,
        'bond_matrices_list': bond_matrices_list
    }

# ======================== Image Encoder ========================

class ImageEncoder(nn.Module):
    """
    Image encoder with ResNet-50 or Swin Transformer backbone.
    Outputs spatial feature maps for Transformer Decoder.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        d_model: int = 512
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            # Remove avgpool and fc
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feat_dim = 2048

        elif backbone.startswith('swin'):
            if backbone == 'swin_t':
                swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_s':
                swin = models.swin_s(weights=models.Swin_S_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_b':
                swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 1024
            elif backbone == 'swin_v2_t':
                swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_v2_s':
                swin = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_v2_b':
                swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 1024
            else:
                raise ValueError(f"Unknown Swin variant: {backbone}")
            
            self.backbone = swin.features
            self.norm = swin.norm
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Project to d_model
        self.proj = nn.Conv2d(self.feat_dim, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB images
        Returns:
            [B, d_model, H', W'] feature maps
        """
        if self.backbone_name == 'resnet50':
            x = self.backbone(x)
        else:
            x = self.backbone(x)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.proj(x)
        return x

# ======================== Positional Encoding ========================

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for image features."""
    
    def __init__(self, d_model: int, max_h: int = 100, max_w: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # d_model must be divisible by 4 (split among y_sin, y_cos, x_sin, x_cos)
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4"
        
        pe = torch.zeros(max_h, max_w, d_model)
        
        y_pos = torch.arange(0, max_h).unsqueeze(1).float()  # [max_h, 1]
        x_pos = torch.arange(0, max_w).unsqueeze(1).float()  # [max_w, 1]
        
        # Each axis gets d_model/4 frequency components
        dim_per_axis = d_model // 4
        div_term = torch.exp(torch.arange(0, dim_per_axis) * -(np.log(10000.0) / dim_per_axis))
        
        # Compute sinusoidal encodings per axis
        y_sin = torch.sin(y_pos * div_term)  # [max_h, dim_per_axis]
        y_cos = torch.cos(y_pos * div_term)  # [max_h, dim_per_axis]
        x_sin = torch.sin(x_pos * div_term)  # [max_w, dim_per_axis]
        x_cos = torch.cos(x_pos * div_term)  # [max_w, dim_per_axis]
        
        # Assemble [max_h, max_w, d_model] encoding table
        for i in range(max_h):
            for j in range(max_w):
                pe[i, j, 0*dim_per_axis:1*dim_per_axis] = y_sin[i]
                pe[i, j, 1*dim_per_axis:2*dim_per_axis] = y_cos[i]
                pe[i, j, 2*dim_per_axis:3*dim_per_axis] = x_sin[j]
                pe[i, j, 3*dim_per_axis:4*dim_per_axis] = x_cos[j]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, H, W]
        Returns:
            x + positional encoding: [B, d_model, H, W]
        """
        B, C, H, W = x.shape
        pe_tensor: torch.Tensor = self.pe
        pe = pe_tensor[:H, :W, :].permute(2, 0, 1).unsqueeze(0)  # [1, d_model, H, W]
        return x + pe


# ======================== KV-Cache Helpers ========================

def _mha_with_kv_cache(
    mha: nn.MultiheadAttention,
    q_input: torch.Tensor,
    kv_new_input: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    cached_v: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-head attention with KV-cache for incremental decoding.

    Projects only the *new* tokens for K/V, concatenates with cached K/V from
    previous steps, and computes attention for the query input.

    Args:
        mha: ``nn.MultiheadAttention`` module (must use packed ``in_proj_weight``).
        q_input:  ``[B, Lq, d_model]`` query input.
        kv_new_input: ``[B, Lnew, d_model]`` new key/value input to project and
                      append to cache, **or** ``None`` to reuse the cache as-is
                      (useful for cross-attention after the first step).
        cached_k: ``[B, n_heads, Lprev, d_head]`` or ``None``.
        cached_v: ``[B, n_heads, Lprev, d_head]`` or ``None``.

    Returns:
        output:  ``[B, Lq, d_model]`` attention output.
        new_k:   ``[B, n_heads, Lprev+Lnew, d_head]`` updated key cache.
        new_v:   ``[B, n_heads, Lprev+Lnew, d_head]`` updated value cache.
    """
    B = q_input.size(0)
    d = mha.embed_dim
    n_heads = mha.num_heads
    d_head = d // n_heads

    W = mha.in_proj_weight  # [3d, d]
    b = mha.in_proj_bias    # [3d] or None

    # --- Q projection (always from q_input) ---
    q = F.linear(q_input, W[:d], b[:d] if b is not None else None)
    q = q.view(B, -1, n_heads, d_head).transpose(1, 2)  # [B, heads, Lq, d_head]

    # --- K / V projection + cache concatenation ---
    if kv_new_input is not None:
        k_new = F.linear(kv_new_input, W[d:2*d], b[d:2*d] if b is not None else None)
        v_new = F.linear(kv_new_input, W[2*d:],  b[2*d:]  if b is not None else None)
        k_new = k_new.view(B, -1, n_heads, d_head).transpose(1, 2)
        v_new = v_new.view(B, -1, n_heads, d_head).transpose(1, 2)
        k = torch.cat([cached_k, k_new], dim=2) if cached_k is not None else k_new
        v = torch.cat([cached_v, v_new], dim=2) if cached_v is not None else v_new
    else:
        assert cached_k is not None, "Must provide kv_new_input or cached_k/v"
        k, v = cached_k, cached_v

    # --- Scaled dot-product attention ---
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)                              # [B, heads, Lq, d_head]
    out = out.transpose(1, 2).contiguous().view(B, -1, d)    # [B, Lq, d]
    out = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

    return out, k, v


# ======================== Sequence Decoder (Transformer) ========================

class SequenceDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for chartok_coords token prediction.

    Consumes encoder memory (image features) and generates the full
    SMILES-with-coordinates token sequence.  Causal-mask caching and
    optional gradient checkpointing are supported for efficiency.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Token embedding (shared for all vocab tokens)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for sequence
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head: predict next token in vocabulary
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Cache for causal masks to avoid recreation every forward pass
        self._causal_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
    
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Get cached causal mask or create and cache a new one."""
        cache_key = (T, device)
        if cache_key not in self._causal_mask_cache:
            self._causal_mask_cache[cache_key] = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device),
                diagonal=1
            )
        return self._causal_mask_cache[cache_key]
    
    def _decoder_forward(self, tgt_emb: torch.Tensor, memory: torch.Tensor, 
                         tgt_mask: torch.Tensor, tgt_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Wrapper for decoder forward pass, used for gradient checkpointing."""
        return self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
    
    def forward(
        self,
        img_features: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_features: [B, d_model, H, W] from ImageEncoder
            tgt_tokens: [B, T] target token indices (teacher forcing)
            tgt_key_padding_mask: [B, T] padding mask (for variable length)
        
        Returns:
            hidden_states: [B, T, d_model] - decoder hidden states
            logits: [B, T, vocab_size] - predictions for each position
        """
        B, C, H, W = img_features.shape
        T = tgt_tokens.size(1)
        device = tgt_tokens.device
        
        # Flatten image features to sequence: [B, H*W, d_model]
        memory = img_features.flatten(2).permute(0, 2, 1)  # [B, H*W, d_model]
        
        # Embed target tokens
        tgt_emb = self.token_embedding(tgt_tokens)  # [B, T, d_model]
        
        # Add positional encoding (optimized: avoid expand by using broadcasting)
        positions = torch.arange(T, device=device)
        tgt_emb = tgt_emb + self.pos_encoder(positions)  # broadcasts [T, d] to [B, T, d]

        # Get cached causal mask
        tgt_mask = self._get_causal_mask(T, device)
        
        # Transformer Decoder with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            # Gradient checkpointing saves memory by recomputing activations during backward
            output = checkpoint(
                self._decoder_forward,
                tgt_emb, memory, tgt_mask, tgt_key_padding_mask,
                use_reentrant=False
            )
        else:
            output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )  # [B, T, d_model]
        
        # Predict next token
        logits = self.output_head(output)  # [B, T, vocab_size]
        
        return output, logits

    def forward_step_cached(
        self,
        memory: torch.Tensor,
        new_token_ids: torch.Tensor,
        step_idx: int,
        cache: Optional[List[Dict[str, Optional[torch.Tensor]]]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Optional[torch.Tensor]]]]:
        """One-step forward through the decoder with a KV-cache.

        Instead of re-encoding the full growing sequence at every step
        (*O(T²)* per step, *O(T³)* total), this method processes only the
        **new token** and reuses projected K/V tensors from earlier steps
        (*O(T)* per step, *O(T²)* total).

        Args:
            memory: ``[B, S, d_model]`` flattened encoder features (compute
                    once via ``img_features.flatten(2).permute(0,2,1)``).
            new_token_ids: ``[B]`` token indices for the current step.
            step_idx: 0-based decoding step index.
            cache: list of per-layer dicts with keys ``self_k``, ``self_v``,
                   ``cross_k``, ``cross_v`` (each ``[B, heads, T, d_head]``
                   or ``None``).  Pass ``None`` on the first call.

        Returns:
            logits: ``[B, vocab_size]`` next-token logits.
            new_cache: updated cache (same structure, one dict per layer).
        """
        B = new_token_ids.size(0)
        device = new_token_ids.device

        # Embed the single new token  →  [B, 1, d_model]
        pos = torch.full((B, 1), step_idx, dtype=torch.long, device=device)
        h = self.token_embedding(new_token_ids.unsqueeze(1)) + self.pos_encoder(pos)

        num_layers = len(self.transformer_decoder.layers)
        if cache is None:
            cache = [
                {'self_k': None, 'self_v': None, 'cross_k': None, 'cross_v': None}
                for _ in range(num_layers)
            ]

        new_cache: List[Dict[str, Optional[torch.Tensor]]] = []
        for i, layer in enumerate(self.transformer_decoder.layers):
            lc = cache[i]

            # --- Self-attention (post-norm, Q=new token, K/V grows) ---
            sa_out, sa_k, sa_v = _mha_with_kv_cache(
                layer.self_attn, h, h, lc['self_k'], lc['self_v'])
            h = layer.norm1(h + layer.dropout1(sa_out))

            # --- Cross-attention (K/V=memory, projected once at step 0) ---
            cross_new = memory if lc['cross_k'] is None else None
            ca_out, ca_k, ca_v = _mha_with_kv_cache(
                layer.multihead_attn, h, cross_new, lc['cross_k'], lc['cross_v'])
            h = layer.norm2(h + layer.dropout2(ca_out))

            # --- Feed-forward ---
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm3(h + layer.dropout3(ff))

            new_cache.append({
                'self_k': sa_k, 'self_v': sa_v,
                'cross_k': ca_k, 'cross_v': ca_v,
            })

        # Final norm (present only if TransformerDecoder was built with one)
        if self.transformer_decoder.norm is not None:
            h = self.transformer_decoder.norm(h)

        logits = self.output_head(h.squeeze(1))  # [B, vocab_size]
        return logits, new_cache


# ======================== Bond Predictor ========================

class BondPredictor(nn.Module):
    """
    Pairwise MLP bond-type predictor.

    Takes hidden states from the SequenceDecoder at atom positions,
    forms all ordered (i, j) pairs via concatenation [h_i, h_j], and
    classifies into *n_bond_classes* bond types.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_bond_classes: int = 7,
    ):
        """
        Args:
            d_model: Hidden dimension from SequenceDecoder.
            n_bond_classes: Number of bond types (0–6: none, single, double, triple, aromatic, wedge-solid, wedge-dash).
        """
        super().__init__()
        self.d_model = d_model
        self.n_bond_classes = n_bond_classes
        
        # MLP for pairwise bond prediction
        # Input: concatenated pair features [2*d_model]
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_bond_classes)
        )
        
        # Cache for diagonal mask to avoid recreation
        self._diag_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
    
    def _get_diag_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """Get cached diagonal mask or create new one."""
        cache_key = (N, device)
        if cache_key not in self._diag_mask_cache:
            self._diag_mask_cache[cache_key] = torch.eye(N, device=device, dtype=torch.bool)
        return self._diag_mask_cache[cache_key]
    
    def _forward_chunk(
        self,
        hidden_states: torch.Tensor,
        atom_indices: torch.Tensor,
        atom_mask: torch.Tensor,
        N: int,
        T: int,
        dim: int,
    ) -> torch.Tensor:
        """Process a chunk of the batch through pair MLP + masking."""
        B_chunk = hidden_states.size(0)
        device = hidden_states.device

        expanded_indices = atom_indices.unsqueeze(-1).expand(B_chunk, N, dim).contiguous()
        atom_hidden = torch.gather(hidden_states, 1, expanded_indices)  # [B_chunk, N, d]

        atom_i = atom_hidden.unsqueeze(2)  # [B_chunk, N, 1, d]
        atom_j = atom_hidden.unsqueeze(1)  # [B_chunk, 1, N, d]
        pair_features = torch.cat(
            [atom_i.expand(-1, -1, N, -1), atom_j.expand(-1, N, -1, -1)], dim=3
        )  # [B_chunk, N, N, 2d]

        edge_logits = self.mlp(pair_features).permute(0, 3, 1, 2)  # [B_chunk, n_bond_classes, N, N]

        valid_pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)  # [B_chunk, N, N]
        diag_mask = self._get_diag_mask(N, device)
        valid_pair_mask = valid_pair_mask & ~diag_mask.unsqueeze(0)
        edge_logits = edge_logits.masked_fill(~valid_pair_mask.unsqueeze(1), -1e4)

        return edge_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        atom_indices: torch.Tensor,
        atom_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with adaptive chunked pairwise computation.

        When the pairwise tensor [B, N, N, 2d] would be very large, the batch
        is split into smaller chunks along dim-0 to cap peak GPU memory.  This
        does NOT change the numerics — gradient flows through ``torch.cat``.
        
        Args:
            hidden_states: [B, T, d_model]
            atom_indices: [B, N] - positions of atom embeddings
            atom_mask: [B, N] - True for valid atoms
        
        Returns:
            edge_logits: [B, n_bond_classes, N, N]
        """
        B, T, dim = hidden_states.shape
        N = atom_indices.size(1)
        device = hidden_states.device
        
        # Ensure tensors are contiguous and on correct device
        atom_indices = atom_indices.to(device).contiguous()
        atom_mask = atom_mask.to(device).contiguous()
        
        # Clamp indices defensively
        atom_indices = atom_indices.clamp(0, T - 1)

        # Adaptive chunking: cap peak pair-tensor memory per chunk.
        # Elements per sample ≈ N*N*2d (concat) + N*N*d (linear hidden) = N*N*3d.
        elements_per_sample = N * N * dim * 3  # conservative estimate
        # 256M elements ≈ 512 MB fp16 — keeps worst-case (N=100) to ~5 chunks
        max_elements = 256 * 1024 * 1024
        chunk_size = max(1, max_elements // max(elements_per_sample, 1))
        chunk_size = min(chunk_size, B)

        if chunk_size >= B:
            # Fast path: entire batch fits comfortably
            return self._forward_chunk(hidden_states, atom_indices, atom_mask, N, T, dim)

        # Chunked path: iterate over sub-batches
        edge_logits_list = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk_logits = self._forward_chunk(
                hidden_states[start:end],
                atom_indices[start:end],
                atom_mask[start:end],
                N, T, dim,
            )
            edge_logits_list.append(chunk_logits)
        return torch.cat(edge_logits_list, dim=0)


# ======================== Helper: Symmetrize Edge Predictions ========================

def _symmetrize_edge_predictions(edge_logits: torch.Tensor) -> np.ndarray:
    r"""Symmetrize bond-type predictions via bidirectional probability averaging.

    Follows MolScribe's ``get_edge_prediction`` logic:
      - Bond types 0–4 (no-bond, single, double, triple, aromatic) are symmetric:
        $p_{ij}^k \leftarrow (p_{ij}^k + p_{ji}^k) / 2$
      - Bond types 5 & 6 (solid-wedge / dash-wedge) are directional, so they
        are cross-averaged:
        $p_{ij}^5 \leftarrow (p_{ij}^5 + p_{ji}^6) / 2$  (and vice-versa)

    Args:
        edge_logits: [n_bond_classes, N, N] raw logits from BondPredictor
                     (single sample, no batch dim).
    Returns:
        edge_preds: [N, N] numpy int array of predicted bond types.
    """
    # Convert logits to probabilities: [N, N, n_bond_classes]
    prob = F.softmax(edge_logits.permute(1, 2, 0).float(), dim=2)

    # Symmetric bond types (0–4): average p[i,j] and p[j,i]
    sym = prob[:, :, :5]
    prob[:, :, :5] = (sym + sym.transpose(0, 1)) / 2

    # Directional wedge bonds (5 & 6): cross-average
    old_5 = prob[:, :, 5].clone()
    old_6 = prob[:, :, 6].clone()
    prob[:, :, 5] = (old_5 + old_6.T) / 2
    prob[:, :, 6] = (old_6 + old_5.T) / 2

    return prob.argmax(dim=2).cpu().numpy()

def extract_atom_indices_from_tokens(
    tgt_tokens: torch.Tensor, 
    vocab: MolScannerVocab
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized extraction of atom positions from chartok_coords sequences.

    In the format [... atom_chars, X_BIN, Y_BIN ...], each atom is anchored
    at its Y_BIN token.  This function returns those positions.

    Args:
        tgt_tokens: [B, T] token-index tensor.
    Returns:
        atom_indices: [B, max_N] positions of Y_BIN tokens (zero-padded).
        atom_counts:  [B] number of atoms per sequence.
    """
    B, T = tgt_tokens.shape
    device = tgt_tokens.device
    
    # Vectorized: find all Y_BIN tokens at once
    is_y_bin = (tgt_tokens >= vocab.y_bin_start_idx) & (tgt_tokens <= vocab.y_bin_end_idx)
    
    # Count atoms per sequence
    atom_counts = is_y_bin.sum(dim=1)  # [B]
    max_atoms = int(atom_counts.max().item())
    
    if max_atoms == 0:
        return torch.zeros((B, 1), dtype=torch.long, device=device), torch.zeros(B, dtype=torch.long, device=device)
    
    # Pre-allocate output tensor
    atom_indices = torch.zeros((B, max_atoms), dtype=torch.long, device=device)
    
    # Vectorized extraction using nonzero
    y_bin_positions = torch.nonzero(is_y_bin, as_tuple=False)  # [num_y_bins, 2]
    
    if y_bin_positions.numel() > 0:
        batch_indices = y_bin_positions[:, 0]
        positions = y_bin_positions[:, 1]
        
        ones = torch.ones(y_bin_positions.size(0), dtype=torch.long, device=device)
        batch_cumsum = torch.zeros(B + 1, dtype=torch.long, device=device)
        batch_cumsum.scatter_add_(0, batch_indices + 1, ones)
        batch_cumsum = batch_cumsum.cumsum(0)
        
        global_idx = torch.arange(y_bin_positions.size(0), device=device)
        in_batch_idx = global_idx - batch_cumsum[batch_indices]
        
        atom_indices[batch_indices, in_batch_idx] = positions
    
    return atom_indices, atom_counts

# ======================== CropWhite for Inference ========================

class AdaptiveLongestMaxSize(A.ImageOnlyTransform):
    """LongestMaxSize with adaptive interpolation.
    
    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    """
    
    def __init__(self, max_size, p=1.0):
        super().__init__(p=p)
        self.max_size = max_size

    def apply(self, img, **params):
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest == self.max_size:
            return img
        scale = self.max_size / longest
        new_h = int(h * scale)
        new_w = int(w * scale)
        interpolation = cv2.INTER_AREA if longest > self.max_size else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    def get_transform_init_args_names(self):
        return ('max_size',)


class AdaptiveResize(A.DualTransform):
    """Resize with adaptive interpolation.
    
    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    Supports keypoint tracking (DualTransform).
    """

    def __init__(self, height, width, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]
        interpolation = cv2.INTER_AREA if (h > self.height or w > self.width) else cv2.INTER_LINEAR
        return {
            "scale_x": self.width / w,
            "scale_y": self.height / h,
            "interpolation": interpolation,
        }

    def apply(self, img, scale_x=1, scale_y=1, interpolation=cv2.INTER_LINEAR, **params):
        return cv2.resize(img, (self.width, self.height), interpolation=interpolation)

    def apply_to_keypoints(self, keypoints, scale_x=1, scale_y=1, **params):
        if keypoints.size == 0:
            return keypoints
        result = keypoints.copy()
        result[:, 0] *= scale_x
        result[:, 1] *= scale_y
        return result

    def get_transform_init_args_names(self):
        return ('height', 'width')


class CropWhiteInference(A.ImageOnlyTransform):
    """Crop white borders from images during inference.
    
    Finds the bounding box of non-white pixels and crops the image,
    keeping a small padding. This removes wasted white space so that
    the molecule content fills more of the final resized image.
    """
    
    def __init__(self, value=(255, 255, 255), pad=5, p=1.0):
        super().__init__(p=p)
        self.value = value
        self.pad = pad
    
    def apply(self, img, **params):
        height, width = img.shape[:2]
        # Find non-white pixels
        if img.ndim == 3:
            non_white = (img != self.value).any(axis=2)
        else:
            non_white = (img != 255)
        
        if not non_white.any():
            return img
        
        # Find bounding box of non-white region
        rows = non_white.any(axis=1)
        cols = non_white.any(axis=0)
        top = rows.argmax()
        bottom = height - rows[::-1].argmax()
        left = cols.argmax()
        right = width - cols[::-1].argmax()
        
        # Crop with padding
        top = max(0, top - self.pad)
        bottom = min(height, bottom + self.pad)
        left = max(0, left - self.pad)
        right = min(width, right + self.pad)
        
        return img[top:bottom, left:right]
    
    def get_transform_init_args_names(self):
        return ('value', 'pad')

# ======================== MolScribeModel ========================

class MolScribeModel(nn.Module):
    """Complete MolScribe model: Image → Token Sequence → Bond Matrix.

    Architecture:
      ImageEncoder  (Swin / ResNet)  →  2-D positional encoding
      SequenceDecoder (Transformer)  →  chartok_coords token logits + hidden states
      BondPredictor   (MLP)          →  pairwise bond-type logits
    """
    
    def __init__(
        self,
        vocab: MolScannerVocab,
        image_size: Tuple[int, int] = (384, 384),
        backbone: str = 'resnet50',
        pretrained: bool = False,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 256*4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.image_size = image_size
        
        # Image Encoder
        self.image_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            d_model=d_model
        )
        self.pos_enc_2d = PositionalEncoding2D(d_model, max_h=20, max_w=20)
        
        # Sequence Decoder (autoregressive Transformer)
        self.sequence_decoder = SequenceDecoder(
            vocab_size=len(vocab),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Bond Predictor (pairwise MLP)
        self.bond_predictor = BondPredictor(
            d_model=d_model,
            n_bond_classes=7,
        )
        
        # Inference transform: crop white borders, resize with padding
        self.inference_transform_list = [
            CropWhiteInference(pad=5),
            AdaptiveLongestMaxSize(max_size=self.image_size[0]),  # INTER_LINEAR↑ / INTER_AREA↓
            A.PadIfNeeded(
                min_height=self.image_size[0], 
                min_width=self.image_size[1], 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=255
            ),
            A.ToGray(num_output_channels=3),
            A.Normalize(),
            ToTensorV2(),
        ]
        self.inference_transforms = A.Compose(self.inference_transform_list)
    
    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        atom_indices: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        max_atoms: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit atom mask."""
        # Defensive shape check for images
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(
                f"Expected images with shape [B, 3, H, W], got {images.shape}. "
                f"Ensure ToGray uses num_output_channels=3."
            )
        
        # Image encoding
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        B, C, H, W = img_features.shape
        
        # Sequence decoding (teacher-forced)
        T = tgt_tokens.size(1)

        max_pos = self.sequence_decoder.pos_encoder.num_embeddings
        if T > max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {max_pos}. "
                f"This should be handled before calling forward()."
            )

        hidden_states, token_logits = self.sequence_decoder(
            img_features=img_features,
            tgt_tokens=tgt_tokens,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # ===== Bond prediction =====
        if atom_indices is None or atom_mask is None or max_atoms is None:
            # Inference mode: generate atom indices and mask from tgt_tokens
            atom_indices, atom_counts = extract_atom_indices_from_tokens(tgt_tokens, self.vocab)
            max_atoms = int(atom_counts.max().item())
            
            if max_atoms == 0:
                edge_logits_padded = torch.zeros((B, 7, 1, 1), device=images.device)
                return token_logits, edge_logits_padded, hidden_states
            
            atom_mask = torch.zeros((B, max_atoms), dtype=torch.bool, device=images.device)
            for i, count in enumerate(atom_counts):
                if count > 0:
                    atom_mask[i, :count] = True
        
        # ===== Align atom_indices tensor to expected max_atoms dimension =====
        current_size = atom_indices.size(1)
        
        if current_size != max_atoms:
            if current_size < max_atoms:
                padding_size = max_atoms - current_size
                atom_indices = F.pad(atom_indices, (0, padding_size), value=0)
                atom_mask = F.pad(atom_mask, (0, padding_size), value=False)
            else:
                atom_indices = atom_indices[:, :max_atoms]
                atom_mask = atom_mask[:, :max_atoms]
        
        atom_indices = atom_indices.to(images.device)
        atom_mask = atom_mask.to(images.device)
        
        # Predict bonds
        if max_atoms == 0:
            edge_logits_padded = torch.zeros((B, 7, 1, 1), device=images.device)
        else:
            T_hidden = hidden_states.size(1)
            atom_indices = atom_indices.clamp(0, T_hidden - 1)
            
            edge_logits_padded = self.bond_predictor(
                hidden_states, 
                atom_indices,
                atom_mask
            )
        
        return token_logits, edge_logits_padded, hidden_states

    def load_model(self, path: str, device: Optional[torch.device] = None):
        """Load trained model weights."""
        if device is None:
            device = next(self.parameters()).device
        
        state_dict = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        
        if is_main_process():
            print(f'Model loaded from: {path}')

# ======================== Inference Functions ========================

    def predict_step(self, img_features: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits given image features and the sequence decoded so far."""
        _, logits = self.sequence_decoder(
            img_features=img_features,
            tgt_tokens=current_tokens,
            tgt_key_padding_mask=None
        )
        return logits[:, -1, :]

    def _apply_constraints(self, logits: torch.Tensor, last_token: int) -> torch.Tensor:
        """Apply structural constraints for chartok_coords format.
        
        Rules:
          - After X_BIN  → only Y_BIN is allowed
          - After Y_BIN  → SMILES chars + EOS (no coords)
          - After SOS    → SMILES chars only (no coords, no EOS)
          - After SMILES char → SMILES chars + X_BIN + EOS (no Y_BIN)
        """
        logits = logits.clone()
        vocab = self.vocab
        
        is_last_x_bin = vocab.is_x_coord_token(last_token)
        is_last_y_bin = vocab.is_y_coord_token(last_token)
        
        if is_last_x_bin:
            # After X_BIN → only Y_BIN allowed
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = False
            logits.masked_fill_(mask, float('-inf'))
            
        elif is_last_y_bin:
            # After Y_BIN → SMILES chars + EOS; no coords
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')
            
        elif last_token == vocab.sos_idx:
            # After SOS → SMILES chars only (no coords, no EOS)
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.eos_idx, vocab.unk_idx]] = float('-inf')
            
        else:
            # After a SMILES char → allow SMILES chars + X_BIN + EOS; no Y_BIN
            logits[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')
            
        return logits

    def _greedy_decode(self, feat: torch.Tensor, max_len: int, device: torch.device) -> List[int]:
        """Greedy decoding for single image."""
        seq = [self.vocab.sos_idx]
        
        for _ in range(max_len):
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            logits = self.predict_step(feat, tgt)[0]
            
            last_token = seq[-1]
            logits = self._apply_constraints(logits, last_token)
            
            next_token = torch.argmax(logits).item()
            seq.append(next_token)
            if next_token == self.vocab.eos_idx:
                break
        
        return seq

    def _beam_search_decode(self, feat: torch.Tensor, beam_size: int, max_len: int, device: torch.device) -> List[int]:
        """Beam search decoding for single image."""
        beams = [([self.vocab.sos_idx], 0.0)]
        completed = []
        
        for _ in range(max_len):
            if not beams:
                break
                
            candidates = []
            for seq, score in beams:
                if seq[-1] == self.vocab.eos_idx:
                    completed.append((seq, score))
                    continue
                
                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.predict_step(feat, tgt)[0]
                
                last_token = seq[-1]
                logits = self._apply_constraints(logits, last_token)
                
                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                
                for prob, idx in zip(topk_probs.tolist(), topk_ids.tolist()):
                    if prob > float('-inf'):
                        candidates.append((seq + [idx], score + prob))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            if len(completed) >= beam_size:
                break
            if completed and beams and beams[0][1] < max(completed, key=lambda x: x[1])[1]:
                break
        
        completed.extend(beams)
        return max(completed, key=lambda x: x[1])[0] if completed else [self.vocab.sos_idx]

    def _postprocess_sequence(self, seq: List[int], feat: torch.Tensor, device: torch.device) -> Dict:
        """Decode a chartok_coords token sequence → SMILES + atom coords, then predict bonds.

        Returns a dict with keys: token_ids, smiles, symbols, coords, bond_mat, success.
        """
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        
        atom_indices, atom_counts = extract_atom_indices_from_tokens(seq_tensor, self.vocab)
        atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
        
        hidden_states, _ = self.sequence_decoder(feat, seq_tensor)
        edge_logits = self.bond_predictor(hidden_states, atom_indices, atom_mask)
        edge_preds = _symmetrize_edge_predictions(edge_logits[0])
        
        # Decode sequence to SMILES + atom symbols/coords
        result = self.vocab.sequence_to_smiles(seq)
        
        return {
            'token_ids': seq,
            'smiles': result.get('smiles', ''),
            'symbols': result.get('symbols', []),
            'coords': result.get('coords', []),
            'bond_mat': edge_preds,
            'success': len(result.get('smiles', '')) > 0
        }

    def _greedy_decode_batch(self, feats: torch.Tensor, max_len: int, device: torch.device) -> List[List[int]]:
        """Batched greedy decoding for multiple images simultaneously.
        
        Instead of B separate decode loops (each with batch=1), runs ONE decoder
        forward pass per step with batch=B.  ~10-30x faster on GPU.
        """
        B = feats.size(0)
        vocab = self.vocab
        
        # All sequences start with SOS
        seqs = torch.full((B, 1), vocab.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            if finished.all():
                break
            
            # Single forward pass for ALL sequences in the batch
            logits = self.predict_step(feats, seqs)  # [B, vocab_size]
            
            # Apply per-sequence constraints (cheap CPU ops, negligible vs decoder)
            for b in range(B):
                if not finished[b]:
                    logits[b] = self._apply_constraints(logits[b], seqs[b, -1].item())
            
            next_tokens = torch.argmax(logits, dim=-1)  # [B]
            
            # For already-finished sequences, emit PAD so they don't affect stats
            next_tokens = torch.where(finished,
                                      torch.full_like(next_tokens, vocab.pad_idx),
                                      next_tokens)
            
            # Mark newly finished
            finished = finished | (next_tokens == vocab.eos_idx)
            
            seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)
        
        # Convert to list-of-lists, trimming after EOS / removing trailing PAD
        result = []
        pad, eos = vocab.pad_idx, vocab.eos_idx
        for b in range(B):
            seq = seqs[b].tolist()
            if eos in seq:
                seq = seq[:seq.index(eos) + 1]
            while seq and seq[-1] == pad:
                seq.pop()
            result.append(seq)
        
        return result

    def _sample_decode_batch(
        self,
        feats: torch.Tensor,
        max_len: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """Batched multinomial sampling with KV-cached decoding.

        Mirrors :meth:`_greedy_decode_batch` but uses temperature-scaled
        multinomial sampling instead of argmax, and replaces the O(T²)-per-
        step full-sequence forward with the O(T)-per-step KV-cached
        :meth:`SequenceDecoder.forward_step_cached`.

        Intended for the RL sampling phase (called under ``torch.no_grad()``).

        Args:
            feats: ``[B, d_model, H, W]`` encoded image features.
            max_len: maximum decoding length.
            device: compute device.
            temperature: softmax temperature (>1 → more exploration).

        Returns:
            List of token-id lists (length B), each starting with SOS and
            ending with EOS (or truncated at *max_len*).
        """
        B = feats.size(0)
        vocab = self.vocab

        # Pre-flatten encoder features → memory [B, S, d_model]
        memory = feats.flatten(2).permute(0, 2, 1)

        tokens = torch.full((B,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)                  # [B, 1] running record
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cache = None

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )  # logits: [B, vocab_size]

            # Per-sequence structural constraints (cheap CPU ops)
            for b in range(B):
                if not finished[b]:
                    logits[b] = self._apply_constraints(
                        logits[b], seqs[b, -1].item()
                    )

            # Temperature-scaled multinomial sampling
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs, 1).squeeze(-1)   # [B]

            tokens = torch.where(
                finished,
                torch.full_like(tokens, vocab.pad_idx),
                tokens,
            )
            finished = finished | (tokens == vocab.eos_idx)
            seqs = torch.cat([seqs, tokens.unsqueeze(1)], dim=1)

        # Convert to list-of-lists, trim to EOS
        result = []
        pad, eos = vocab.pad_idx, vocab.eos_idx
        for b in range(B):
            seq = seqs[b].tolist()
            if eos in seq:
                seq = seq[:seq.index(eos) + 1]
            while seq and seq[-1] == pad:
                seq.pop()
            result.append(seq)

        return result

    def _postprocess_sequences_batch(self, all_seqs: List[List[int]],
                                      img_features: torch.Tensor,
                                      device: torch.device) -> List[Dict]:
        """Batched bond prediction + SMILES reconstruction for decoded sequences.
        
        Pads all sequences, runs one decoder + bond-predictor forward pass,
        then does per-sample CPU post-processing.
        """
        B = len(all_seqs)
        vocab = self.vocab
        
        # Pad sequences to same length
        max_len = max(len(s) for s in all_seqs)
        padded = torch.full((B, max_len), vocab.pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(all_seqs):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        
        padding_mask = (padded == vocab.pad_idx)  # True = padded position
        
        # Extract atom indices (already supports batched input)
        atom_indices, atom_counts = extract_atom_indices_from_tokens(padded, vocab)
        max_atoms = int(atom_counts.max().item())
        
        # Batched decoder forward + bond predictor (single GPU pass)
        edge_logits_all = None
        if max_atoms > 0:
            atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
            hidden_states, _ = self.sequence_decoder(img_features, padded, tgt_key_padding_mask=padding_mask)
            edge_logits_all = self.bond_predictor(hidden_states, atom_indices, atom_mask)
        
        # Per-sample CPU post-processing
        results = []
        for b in range(B):
            seq = all_seqs[b]
            result = vocab.sequence_to_smiles(seq)
            
            if edge_logits_all is not None and atom_counts[b] > 0:
                edge_preds = _symmetrize_edge_predictions(edge_logits_all[b])
            else:
                edge_preds = np.zeros((0, 0), dtype=np.int64)
            
            results.append({
                'token_ids': seq,
                'smiles': result.get('smiles', ''),
                'symbols': result.get('symbols', []),
                'coords': result.get('coords', []),
                'bond_mat': edge_preds,
                'success': len(result.get('smiles', '')) > 0,
            })
        
        return results

    @torch.no_grad()
    def generate(self, images: torch.Tensor, beam_size: int = 1, max_len: int = 500, device: Optional[torch.device] = None) -> List[Dict]:
        """Auto-regressively decode token sequences and predict bond matrices for a batch of images.
        
        When beam_size=1 and B>1, uses batched greedy decoding for much higher
        GPU utilization (~10-30x faster than sequential single-image decoding).
        """
        self.eval()
        
        if device is None:
            device = images.device
        
        B = images.size(0)
        
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        # --- Fast path: batched greedy decode ---
        if beam_size == 1 and B > 1:
            all_seqs = self._greedy_decode_batch(img_features, max_len, device)
            return self._postprocess_sequences_batch(all_seqs, img_features, device)
        
        # --- Fallback: per-image decode (beam search or single image) ---
        results = []
        for b in range(B):
            feat = img_features[b:b+1]
            
            if beam_size == 1:
                seq = self._greedy_decode(feat, max_len, device)
            else:
                seq = self._beam_search_decode(feat, beam_size, max_len, device)
            
            result = self._postprocess_sequence(seq, feat, device)
            results.append(result)
        
        return results

    def _preprocess_tensor(self, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Preprocess tensor input, ensuring correct size and dimensions."""
        img_tensor = img_tensor.to(device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        _, _, h, w = img_tensor.shape
        if (h, w) != self.image_size:
            img_tensor = F.interpolate(img_tensor, size=self.image_size, mode='bilinear', align_corners=False)
        return img_tensor

    def _preprocess_image(self, image_source, device: torch.device) -> torch.Tensor:
        """Preprocess raw image input (path/numpy/PIL)."""
        if isinstance(image_source, str):
            img_pil = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                img_pil = Image.fromarray(image_source).convert('RGB')
            elif image_source.shape[2] == 3:
                img_pil = Image.fromarray(image_source[:, :, ::-1])
            elif image_source.shape[2] == 4:
                img_pil = Image.fromarray(image_source[:, :, :3][:, :, ::-1])
            else:
                raise ValueError(f"Unsupported numpy array shape: {image_source.shape}")
        elif isinstance(image_source, Image.Image):
            img_pil = image_source.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")
        
        return self.inference_transforms(image=np.array(img_pil))['image'].unsqueeze(0).to(device)

    def predict(self, image_source, device: Optional[torch.device] = None, beam_size: int = 3, max_len: int = 500,
                return_preprocessed: bool = False, smiles_mode: Optional[str] = None) -> Dict:
        """End-to-end prediction for a single image.
        
        Args:
            image_source: file path, numpy array, PIL Image, or pre-processed tensor.
            smiles_mode: if set, add 'pred_smiles' key using the given mode.
                One of 'decoder', 'graph', 'postprocess', or None (no conversion).
        """
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(image_source, torch.Tensor):
            img_tensor = self._preprocess_tensor(image_source, device)
        else:
            img_tensor = self._preprocess_image(image_source, device)
        
        result = self.generate(images=img_tensor, beam_size=beam_size, max_len=max_len, device=device)[0]
        
        if return_preprocessed:
            result['preprocessed_img'] = img_tensor.squeeze(0).cpu()
        
        if smiles_mode is not None:
            result['pred_smiles'] = _result_to_smiles(result, mode=smiles_mode)
        
        return result

    def predict_batch(self, image_sources: List, 
                      device: Optional[torch.device] = None, 
                      beam_size: int = 3,
                      max_len: int = 500, 
                      smiles_mode: Optional[str] = None) -> List[Dict]:
        """Batch prediction on a single device.
        
        Args:
            image_sources: list of file paths, numpy arrays, PIL Images, or tensors.
            smiles_mode: if set, add 'pred_smiles' key to each result.
                One of 'decoder', 'graph', 'postprocess', or None.
        """
        if device is None:
            if list(self.parameters()):
                device = next(self.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        if isinstance(device, str):
            device = torch.device(device)

        tensors = []
        cpu_device = torch.device('cpu') 
        
        for src in image_sources:
            if isinstance(src, torch.Tensor):
                tensors.append(self._preprocess_tensor(src, cpu_device))
            else:
                tensors.append(self._preprocess_image(src, cpu_device))
        
        if not tensors:
            return []

        img_batch = torch.cat(tensors, dim=0).to(device)
        results = self.generate(images=img_batch, beam_size=beam_size, max_len=max_len, device=device)
        
        if smiles_mode is not None:
            for r in results:
                r['pred_smiles'] = _result_to_smiles(r, mode=smiles_mode)
        
        return results

# ======================== Loss Computation ========================

def get_bond_class_weights():
    """Get predefined class weights for bond matrix."""
    weights = torch.tensor([
        1.0,   # 0: no bond (very common, lower weight)
        10.0,   # 1: single bond
        10.0,   # 2: double bond
        10.0,   # 3: triple bond (rare)
        10.0,   # 4: aromatic bond
        10.0,   # 5: solid wedge (stereo, rare)
        10.0    # 6: dash wedge (stereo, rare)
    ], dtype=torch.float32)
    
    if is_main_process():
        print(f'Using predefined bond class weights: {weights.numpy()}')
    
    return weights

def compute_losses(
    token_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    tgt_tokens: torch.Tensor,
    bond_matrices_list: List[np.ndarray],
    vocab: MolScannerVocab,
    bond_class_weights: torch.Tensor,
    token_label_smoothing: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute token-prediction and bond-prediction losses.

    Token loss uses label smoothing; bond loss uses per-class weights.
    """
    B, T, vocab_size = token_logits.shape
    device = token_logits.device
    
    # ===== 1. Token loss =====
    token_logits_flat = token_logits.reshape(-1, vocab_size)
    tgt_tokens_flat = tgt_tokens.reshape(-1)
    
    token_loss = F.cross_entropy(
        token_logits_flat, 
        tgt_tokens_flat, 
        ignore_index=vocab.pad_idx,
        label_smoothing=token_label_smoothing,
        reduction='mean'
    )
    
    # ===== 2. Bond loss =====
    B, N_bond_type, N_atom_pred, _ = edge_logits.shape
    
    bond_targets = torch.full((B, N_atom_pred, N_atom_pred), -100, dtype=torch.long, device=device)
    
    for b, bond_mat in enumerate(bond_matrices_list):
        N_atom_gt = bond_mat.shape[0]
        
        if N_atom_gt == 0 or N_atom_gt > N_atom_pred:
            continue
            
        bond_mat_tensor = torch.from_numpy(bond_mat.astype(np.int64)).to(device)
        bond_targets[b, :N_atom_gt, :N_atom_gt] = bond_mat_tensor
        bond_targets[b].fill_diagonal_(-100)

    if (bond_targets != -100).any():
        avg_bond_loss = F.cross_entropy(
            edge_logits,
            bond_targets,
            weight=bond_class_weights.to(device),
            ignore_index=-100,
            reduction='mean'
        )
    else:
        avg_bond_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = token_loss + avg_bond_loss
    
    return {
        'token_loss': token_loss,
        'bond_loss': avg_bond_loss,
        'total_loss': total_loss
    }

# ======================== Sequence Truncation Helper ========================

def truncate_sequences(
    tgt_tokens: torch.Tensor,
    tgt_padding_mask: torch.Tensor,
    atom_indices: torch.Tensor,
    atom_mask: torch.Tensor,
    max_seq_len: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncate sequences to max_seq_len and update atom indices/mask accordingly."""
    if tgt_tokens.size(1) <= max_seq_len:
        effective_max = tgt_tokens.size(1) - 1
        valid_atom_mask = atom_indices < effective_max
        atom_mask = atom_mask & valid_atom_mask
        atom_indices = torch.where(valid_atom_mask, atom_indices, torch.zeros_like(atom_indices)).contiguous()
        return tgt_tokens, tgt_padding_mask, atom_indices, atom_mask.contiguous()
    
    tgt_tokens = tgt_tokens[:, :max_seq_len].contiguous()
    tgt_padding_mask = tgt_padding_mask[:, :max_seq_len].contiguous()
    
    effective_max = max_seq_len - 1
    valid_atom_mask = atom_indices < effective_max
    atom_mask = atom_mask & valid_atom_mask
    atom_indices = torch.where(valid_atom_mask, atom_indices, torch.zeros_like(atom_indices)).contiguous()
    
    return tgt_tokens, tgt_padding_mask, atom_indices, atom_mask.contiguous()

# ======================== Utility Functions ========================

def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0

def remove_atom_mapping(smiles: str) -> str:
    """Remove atom mapping numbers from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles

# ======================== Evaluation & Multi-GPU Inference ========================

def _load_benchmark_gt(benchmark_dir: str, csv_path: str,
                       max_samples: Optional[int] = None) -> List[Dict]:
    """Load benchmark CSV, canonicalize GT SMILES, return list of dicts."""
    label_df = pd.read_csv(csv_path)
    if max_samples is not None and max_samples < len(label_df):
        label_df = label_df.sample(n=max_samples, random_state=42)

    data = []
    for _, row in label_df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(benchmark_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        try:
            gt_smi = remove_atom_mapping(row['SMILES'])
            gt_smi, ok = canonicalize_smiles(gt_smi, ignore_cistrans=True)
        except Exception:
            gt_smi, ok = None, False
        data.append({'image_id': img_id, 'img_path': img_path,
                     'gt_smiles': gt_smi, 'gt_ok': ok})
    return data


def _result_to_smiles_decoder(result: Dict) -> Optional[str]:
    """Mode 1: SMILES directly from decoder sequence → canonicalize."""
    if not result or not result.get('success'):
        return None
    try:
        smiles = result.get('smiles', '')
        if not smiles:
            return None
        can_smi, ok = canonicalize_smiles(smiles, ignore_cistrans=True)
        return can_smi if ok and can_smi else None
    except Exception:
        return None


def _result_to_smiles_graph(result: Dict) -> Optional[str]:
    """Mode 2: SMILES entirely reconstructed from predicted atoms + bonds."""
    if not result or not result.get('success'):
        return None
    try:
        symbols = result.get('symbols', [])
        coords = result.get('coords', [])
        bond_mat = result.get('bond_mat')
        if not symbols or not coords or bond_mat is None:
            return None
        smi, _, _, ok = _convert_graph_to_smiles(
            coords=coords, symbols=symbols, edges=bond_mat)
        if not ok or not smi or smi == '<invalid>':
            return None
        smi, ok = canonicalize_smiles(smi, ignore_cistrans=True)
        return smi if ok else None
    except Exception:
        return None


def _result_to_smiles_postprocess(result: Dict) -> Optional[str]:
    """Mode 3 (MolScribe-style): decoder SMILES + postprocessing.
    
    Follows MolScribe's _postprocess_smiles workflow:
      1. Replace R-groups / unknown tokens with isotope-labeled wildcards
      2. Strip stereo from SMILES, build mol (sanitize=False)
      3. Restore chirality/E-Z via _verify_chirality with predicted coords/edges
      4. Expand functional groups back using the mappings
      5. Return canonical SMILES
    """
    if not result or not result.get('success'):
        return None

    smiles = result.get('smiles', '')
    if not isinstance(smiles, str) or smiles == '':
        return None

    coords = result.get('coords', [])
    symbols = result.get('symbols', [])
    edges = result.get('bond_mat')

    try:
        pred_smiles = smiles
        # Step 1: replace R-groups / abbreviations with placeholders
        pred_smiles, mappings = _replace_functional_group(pred_smiles)

        # Step 2: if we have graph info, strip stereo and restore via coordinates
        if coords and symbols and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, edges)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)

        # Step 3: expand functional groups back
        pred_smiles, mol = _expand_functional_group(mol, mappings)

        if pred_smiles and pred_smiles != '<invalid>':
            # Canonicalize (with ignore_cistrans) to match GT preprocessing
            can_smi, ok = canonicalize_smiles(pred_smiles, ignore_cistrans=True)
            return can_smi if ok and can_smi else None
        return None
    except Exception:
        # Fallback: try plain canonicalize of raw decoder SMILES
        try:
            can_smi, ok = canonicalize_smiles(smiles, ignore_cistrans=True)
            return can_smi if ok else None
        except Exception:
            return None


def _result_to_smiles(result: Dict, mode: str = SMILES_MODE_POSTPROCESS) -> Optional[str]:
    """Dispatcher: convert prediction result → canonical SMILES.
    
    Args:
        result: prediction dict with keys 'smiles', 'symbols', 'coords', 'bond_mat', 'success'.
        mode: one of SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS.
    """
    if mode == SMILES_MODE_DECODER:
        return _result_to_smiles_decoder(result)
    elif mode == SMILES_MODE_GRAPH:
        return _result_to_smiles_graph(result)
    elif mode == SMILES_MODE_POSTPROCESS:
        return _result_to_smiles_postprocess(result)
    else:
        raise ValueError(f"Unknown smiles_mode: {mode!r}. "
                         f"Choose from '{SMILES_MODE_DECODER}', '{SMILES_MODE_GRAPH}', '{SMILES_MODE_POSTPROCESS}'.")

def _compute_benchmark_metrics(gt_data: List[Dict],
                               pred_smiles_list: List[Optional[str]],
                               with_records: bool = False) -> Dict:
    """Compute exact match accuracy and avg Tanimoto from GT/pred SMILES lists."""
    exact_match, failed_gt, failed_pred = 0, 0, 0
    tanimoto_scores = []
    records = [] if with_records else None

    for gt, pred_smi in zip(gt_data, pred_smiles_list):
        if not gt['gt_ok']:
            failed_gt += 1
            if records is not None:
                records.append({'image_id': gt['image_id'], 'gt_smiles': gt['gt_smiles'],
                                'pred_smiles': None, 'match': False, 'tanimoto': 0.0})
            continue
        if pred_smi is None:
            failed_pred += 1
            tanimoto_scores.append(0.0)
            if records is not None:
                records.append({'image_id': gt['image_id'], 'gt_smiles': gt['gt_smiles'],
                                'pred_smiles': None, 'match': False, 'tanimoto': 0.0})
            continue

        match = pred_smi == gt['gt_smiles']
        if match:
            exact_match += 1
        tan = compute_tanimoto_similarity(gt['gt_smiles'], pred_smi)
        tanimoto_scores.append(tan)
        if records is not None:
            records.append({'image_id': gt['image_id'], 'gt_smiles': gt['gt_smiles'],
                            'pred_smiles': pred_smi, 'match': match, 'tanimoto': tan})

    valid = len(gt_data) - failed_gt
    acc = exact_match / valid * 100 if valid > 0 else 0.0
    avg_tan = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0

    out = {'exact_match_acc': acc, 'avg_tanimoto': avg_tan,
           'total': len(gt_data), 'valid': valid, 'failed_predictions': failed_pred}
    if records is not None:
        out['records_df'] = pd.DataFrame(records)
    return out

# ---------- DDP training validation (all ranks participate via all_reduce) ----------

class _ValImageDataset(Dataset):
    """Lightweight dataset for batched validation image loading.

    Returns (image_tensor, index) pairs.  Images that fail to load are
    replaced with zero tensors so that the entire batch is not lost.
    """

    def __init__(self, items: List[Dict], transforms, image_size: Tuple[int, int] = (384, 384)):
        self.items = items
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.items[idx]['img_path']).convert('RGB')
            tensor = self.transforms(image=np.array(img))['image']
        except Exception:
            # Fallback: black image — will produce garbage output, counted as failed
            tensor = torch.zeros(3, *self.image_size)
        return tensor, idx


def validate(
    model: nn.Module,
    benchmark_dir: str,
    benchmark_csv_path: str,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    global_step: int,
    beam_size: int = 1,
    max_samples: Optional[int] = None,
    val_batch_size: int = 128,
) -> Dict[str, float]:
    """Evaluate on benchmark using ALL DDP ranks; metrics aggregated via all_reduce.

    Uses **mini-batched inference** with parallel image loading for high GPU
    utilisation (~80-95% vs ~20-25% with single-image processing).

    Speed gains come from three layers:
      1. DataLoader with num_workers for overlapped CPU image decoding / transforms.
      2. Batched Swin-B encoder forward pass (128 images at once).
      3. Batched greedy decoding — one decoder step processes all B sequences in
         parallel instead of B separate loops with batch=1.

    Reports accuracy for all three SMILES modes:
      - decoder:     SMILES directly from sequence decoding
      - graph:       SMILES reconstructed from predicted atoms + bonds
      - postprocess: decoder SMILES + chirality correction via coords/edges
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    rank, world_size = get_rank(), get_world_size()

    gt_data = _load_benchmark_gt(benchmark_dir, benchmark_csv_path, max_samples)
    # For DDP training val, skip entries with bad GT to keep things clean
    gt_data = [d for d in gt_data if d['gt_ok']]

    if is_main_process():
        print(f'\nEvaluating on {benchmark_csv_path} ({len(gt_data)} valid, {world_size} GPUs)')
    if not gt_data:
        return {'exact_match_acc': 0.0, 'avg_tanimoto': 0.0, 'valid_samples': 0, 'failed_predictions': 0}

    my_data = gt_data[rank::world_size]

    modes = [SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS]
    # Per-mode counters: exact, failed, tan_sum
    local_stats = {m: [0, 0, 0.0] for m in modes}  # [exact, failed, tan_sum]
    local_count = 0

    # --- Parallel image loading via DataLoader ---
    val_dataset = _ValImageDataset(
        my_data, actual_model.inference_transforms, actual_model.image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        n_batches = len(val_loader)
        it = (tqdm(val_loader, total=n_batches, desc=f'Epoch {epoch} [Val R{rank}]')
              if is_main_process() else val_loader)

        for img_batch, indices in it:
            img_batch = img_batch.to(device, non_blocking=True)
            B_cur = img_batch.size(0)
            local_count += B_cur

            # --- Batched forward: encoder + greedy decode + bond predictor ---
            try:
                batch_results = actual_model.generate(
                    images=img_batch, beam_size=beam_size, device=device)
            except Exception:
                batch_results = [None] * B_cur

            # --- Per-sample SMILES evaluation (CPU-bound, ~2-5 ms each) ---
            for local_b in range(B_cur):
                idx = indices[local_b].item()
                gt_smi = my_data[idx]['gt_smiles']
                result = batch_results[local_b]

                for mode in modes:
                    try:
                        pred_smi = _result_to_smiles(result, mode=mode) if result else None
                    except Exception:
                        pred_smi = None
                    if pred_smi is None:
                        local_stats[mode][1] += 1  # failed
                        continue
                    if pred_smi == gt_smi:
                        local_stats[mode][0] += 1  # exact
                    local_stats[mode][2] += compute_tanimoto_similarity(gt_smi, pred_smi)

    # Pack: [count, dec_exact, dec_failed, dec_tan, graph_exact, graph_failed, graph_tan, post_exact, post_failed, post_tan]
    flat = [float(local_count)]
    for mode in modes:
        flat.extend([float(local_stats[mode][0]), float(local_stats[mode][1]), local_stats[mode][2]])
    stats = torch.tensor(flat, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_count = int(stats[0])
    results_out = {}

    if is_main_process():
        print(f'Epoch {epoch} [Val] ({total_count} samples):')

    for i, mode in enumerate(modes):
        exact = int(stats[1 + i * 3])
        failed = int(stats[2 + i * 3])
        tan_sum = stats[3 + i * 3].item()
        acc = exact / total_count if total_count else 0.0
        avg_tan = tan_sum / total_count if total_count else 0.0

        results_out[f'{mode}/exact_match_acc'] = acc
        results_out[f'{mode}/avg_tanimoto'] = avg_tan
        results_out[f'{mode}/failed'] = failed

        if is_main_process():
            print(f'  [{mode:11s}] Exact={acc:.4f} ({exact}/{total_count})  '
                  f'Tanimoto={avg_tan:.4f}  Failed={failed}')
            if writer:
                writer.add_scalar(f'Val/{mode}_exact_match_acc', acc, global_step)
                writer.add_scalar(f'Val/{mode}_avg_tanimoto', avg_tan, global_step)
                writer.add_scalar(f'Val/{mode}_failed', failed, global_step)

    # Also store default (postprocess) under canonical keys for backward compat
    results_out['exact_match_acc'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/exact_match_acc', 0.0)
    results_out['avg_tanimoto'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/avg_tanimoto', 0.0)
    results_out['valid_samples'] = total_count
    results_out['failed_predictions'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/failed', 0)

    return results_out

# ---------- Multi-GPU inference via mp.spawn (for standalone / notebook use) ----------

def _inference_worker(rank: int, world_size: int, model: MolScribeModel,
                      data_paths: List[str], return_dict: dict,
                      beam_size: int = 1, mini_batch_size: int = 128):
    """Worker for mp.spawn — runs inference on one GPU slice."""
    cv2.setNumThreads(0)
    try:
        device = torch.device(f'cuda:{rank}')
        chunk = math.ceil(len(data_paths) / world_size)
        my_paths = data_paths[rank * chunk:(rank + 1) * chunk]
        if not my_paths:
            return
        model.to(device); model.eval()
        results = []
        it = range(0, len(my_paths), mini_batch_size)
        if rank == 0:
            it = tqdm(it, desc=f"GPU-{rank}", total=math.ceil(len(my_paths) / mini_batch_size))
        for i in it:
            results.extend(model.predict_batch(my_paths[i:i + mini_batch_size],
                                               device=device, beam_size=beam_size))
        return_dict[rank] = results
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Rank {rank} failed: {e}")


def predict_multigpu(model: MolScribeModel, image_paths: List[str],
                     beam_size: int = 1) -> List[Dict]:
    """Distribute inference across all GPUs via mp.spawn. Returns ordered results."""
    world_size = torch.cuda.device_count()
    if world_size == 0:
        return model.predict_batch(image_paths, beam_size=beam_size)

    print(f"Distributing {len(image_paths)} images across {world_size} GPUs...")
    model.cpu(); model.share_memory()
    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(_inference_worker,
                 args=(world_size, model, image_paths, return_dict, beam_size),
                 nprocs=world_size, join=True)
    except Exception as e:
        print(f"mp.spawn failed: {e}"); return []

    results, chunk = [], math.ceil(len(image_paths) / world_size)
    for rank in range(world_size):
        if rank in return_dict:
            results.extend(return_dict[rank])
        else:
            results.extend([{'success': False}] * min(chunk, len(image_paths) - len(results)))
    return results

def evaluate_benchmarks(
    model: MolScribeModel,
    benchmarks: List[Dict],
    beam_size: int = 1,
    postproc_workers: int = 32,
) -> Dict[str, Dict]:
    """
    Evaluate model on multiple benchmarks using all GPUs (mp.spawn).
    Reports all three SMILES modes (decoder / graph / postprocess) per benchmark.

    Args:
        model: Loaded MolScribeModel.
        benchmarks: List of dicts, each with keys 'name', 'benchmark_dir', 'csv_path'.
        beam_size: Beam width for decoding (1 = greedy).
        postproc_workers: Thread-pool size for parallel SMILES post-processing.

    Returns:
        Dict mapping benchmark name → {<mode>/exact_match_acc, <mode>/avg_tanimoto, …}
    """
    from concurrent.futures import ProcessPoolExecutor
    all_results = {}
    modes = [SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS]

    for b in benchmarks:
        name = b['name']
        print(f"\n{'='*50}\nBenchmark: {name}\n{'='*50}")

        gt_data = _load_benchmark_gt(b['benchmark_dir'], b['csv_path'])
        image_paths = [d['img_path'] for d in gt_data]
        print(f"  Images: {len(image_paths)}")

        raw_results = predict_multigpu(model, image_paths, beam_size=beam_size)
        min_len = min(len(raw_results), len(gt_data))
        raw_results, gt_data = raw_results[:min_len], gt_data[:min_len]

        benchmark_stats = {}
        for mode in modes:
            print(f"  Post-processing [{mode}] ...")
            converter = partial(_result_to_smiles, mode=mode)
            chunksize = max(1, len(raw_results) // (postproc_workers * 4))
            with ProcessPoolExecutor(max_workers=postproc_workers) as ex:
                pred_smiles = list(tqdm(ex.map(converter, raw_results, chunksize=chunksize),
                                        total=len(raw_results), desc=f"  {mode}"))

            stats = _compute_benchmark_metrics(gt_data, pred_smiles, with_records=(mode == SMILES_MODE_POSTPROCESS))
            print(f"  [{name}/{mode:11s}] Exact Match: {stats['exact_match_acc']:.2f}% "
                  f"Tanimoto: {stats['avg_tanimoto']:.4f}  Failed: {stats['failed_predictions']}")
            for k, v in stats.items():
                benchmark_stats[f'{mode}/{k}'] = v

        benchmark_stats['total'] = len(gt_data)
        all_results[name] = benchmark_stats

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Benchmark':<12} {'Mode':<13} {'Exact Match':>12} {'Tanimoto':>10} {'Failed':>8}")
    print(f"{'-'*80}")
    for name, d in all_results.items():
        for mode in modes:
            acc = d.get(f'{mode}/exact_match_acc', 0)
            tan = d.get(f'{mode}/avg_tanimoto', 0)
            fail = d.get(f'{mode}/failed_predictions', 0)
            print(f"{name:<12} {mode:<13} {acc:>11.2f}% {tan:>10.4f} {fail:>8}")
    print(f"{'='*80}")
    return all_results

# ======================== DDP Training Loop ========================

def train(
        *,
        # data (PubChem SMILES — dynamically rendered synthetic images)
        pubchem_smiles_list: Optional[List[str]] = None,
        pubchem_smiles_num: Optional[int] = None,
        # data (USPTO patent molecule images — real-world file-based)
        uspto_csv_path: Optional[str] = None,
        uspto_data_dir: Optional[str] = None,
        # training 
        save_path: str,
        num_epochs: int,
        batch_size: int,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float,
        warmup_ratio: float,
        seed: int,
        early_stopping_patience: int,
        # Validation
        benchmark_dir: str,
        benchmark_csv_path: str,
        val_max_samples: Optional[int] = None,
        # Molecule
        max_atoms: int = 100,
        mol_augment: bool = True,
        # Vision encoder
        image_size: Tuple[int, int] = (384, 384),
        n_bins: int = 64,
        backbone: str = 'swin_b',
        pretrained: bool = True,
        # decoder
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        # ===== optimization parameters =====
        num_workers: int = 4,
        use_amp: bool = True,
        use_gradient_checkpointing: bool = False,
        force_cpu: bool = False,
        # ===== resume from checkpoint =====
        resume_from: Optional[str] = None,
        # ===== fine-tune from pretrained weights (model only, fresh optimizer) =====
        finetune_from: Optional[str] = None,
) -> List[float]:
    """
    DDP training loop with cosine-annealed LR, AMP, and per-epoch validation.

    Launch with torchrun:
        torchrun --nproc_per_node=<N_GPUS> train_MolScanner_ddp.py

    Supports three data configurations:
      1. PubChem synthetic only:  pass *pubchem_smiles_list* + *pubchem_smiles_num*
      2. USPTO only:              pass *uspto_csv_path* + *uspto_data_dir*
      3. Joint (recommended):     pass both PubChem and USPTO args

    Args:
        pubchem_smiles_list: Pool of PubChem SMILES for synthetic rendering.
        pubchem_smiles_num: Number of SMILES to sample from the pool.
        uspto_csv_path: Path to USPTO CSV with file_path/SMILES/node_coords/edges.
        uspto_data_dir: Root directory for resolving image paths in USPTO CSV.
        save_path: Directory for checkpoints and TensorBoard logs.
        benchmark_dir / benchmark_csv_path: Validation benchmark.
        batch_size: **Per-GPU** batch size (effective = batch_size × world_size).
        (remaining args are self-explanatory model / optimization hyper-parameters)
    """
    # ===== DDP setup =====
    # torchrun sets LOCAL_RANK / WORLD_SIZE automatically
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if force_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        use_amp = False
        if is_main_process():
            print('Forcing CPU mode')
    else:
        # Initialize DDP
        if world_size > 1:
            setup_ddp(local_rank, world_size)
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    
    if is_main_process():
        if world_size > 1:
            print(f'DDP Training: {world_size} GPUs')
            for i in range(world_size):
                print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print(f'Single GPU training on: {device}')
        effective_batch = batch_size * world_size
        print(f'Per-GPU batch size: {batch_size}, Effective batch size: {effective_batch}')
        print(f'Optimizations: AMP={use_amp}, GradCheckpoint={use_gradient_checkpointing}')
        print(f'Optimizer: encoder_lr={encoder_lr}, decoder_lr={decoder_lr}, weight_decay={weight_decay}')
        print(f'cudnn.benchmark={torch.backends.cudnn.benchmark}, cudnn.deterministic={torch.backends.cudnn.deterministic}')

    # Set seed (different per rank for data augmentation diversity, same for weight init)
    torch.manual_seed(seed)
    np.random.seed(seed + local_rank)  # Different augmentation per GPU
    random.seed(seed + local_rank)
    
    # Create vocab
    vocab = MolScannerVocab(n_bins=n_bins)
    if is_main_process():
        print(f'Vocab size: {len(vocab)}')
    
    # Prepare training data
    has_pubchem = pubchem_smiles_list is not None
    has_uspto = uspto_csv_path is not None
    if not has_pubchem and not has_uspto:
        raise ValueError('Must provide pubchem_smiles_list and/or uspto_csv_path')

    datasets = []  # list of (name, dataset) tuples

    if has_uspto:
        if is_main_process():
            print(f'USPTO data: {uspto_csv_path}')
        uspto_dataset = USPTOMolDataset(
            vocab=vocab,
            csv_path=uspto_csv_path,
            data_dir=uspto_data_dir or '',
            image_size=image_size,
            img_augment=True,
            geo_augment=True,
        )
        datasets.append(('USPTO', uspto_dataset))
        if is_main_process():
            print(f'USPTO dataset size: {len(uspto_dataset)}')

    if has_pubchem:
        n = len(pubchem_smiles_list)
        rng = np.random.default_rng(seed)  # Same seed for all ranks -> same data selection
        indices = np.arange(n)
        rng.shuffle(indices)
        if pubchem_smiles_num is not None:
            indices = indices[:pubchem_smiles_num]
        sampled_smiles = [pubchem_smiles_list[i] for i in indices]
        if is_main_process():
            print(f'PubChem synthetic data: {len(sampled_smiles)} SMILES')
        pubchem_dataset = MoleculeDataset(
            smiles_list=sampled_smiles,
            shuffle_smiles=True,
            vocab=vocab,
            image_size=image_size,
            mol_augment=mol_augment,
            geo_augment=True,
            img_augment=True,
        )
        datasets.append(('PubChem', pubchem_dataset))

    if len(datasets) > 1:
        train_dataset = ConcatDataset([d for _, d in datasets])
        if is_main_process():
            parts = ' + '.join(f'{name}: {len(d)}' for name, d in datasets)
            print(f'Joint training dataset: {len(train_dataset)} ({parts})')
    else:
        train_dataset = datasets[0][1]
        if is_main_process():
            print(f'Train size: {len(train_dataset)} ({datasets[0][0]} only)')

    if is_main_process():
        print(f'Benchmark validation: {benchmark_csv_path}')
    
    # Get predefined bond class weights
    bond_class_weights = get_bond_class_weights()
    
    # ===== DistributedSampler =====
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=True,
        seed=seed
    ) if world_size > 1 else None
    
    # DataLoader - batch_size is PER-GPU
    # Use forkserver to avoid segfaults from fork-unsafe libs (RDKit, OpenCV)
    dl_common = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'collate_fn': partial(collate_fn, vocab=vocab, max_atoms_limit=max_atoms),
        'shuffle': (train_sampler is None),  # Only shuffle if no sampler
        'sampler': train_sampler,
        'drop_last': True,  # Avoid uneven batch sizes across GPUs
    }
    if num_workers > 0:
        dl_common.update({
            'prefetch_factor': 4,
            'persistent_workers': False,  # Allow stuck workers to be cleaned up between epochs
            'multiprocessing_context': 'forkserver',
            'timeout': 60,    # 1 min: fast detection of C-extension deadlocks (func_set_timeout has 5s first)
        })
    
    def _make_loader():
        return DataLoader(train_dataset, **dl_common)
    
    train_loader = _make_loader()
    
    # Model
    model = MolScribeModel(
        vocab=vocab,
        backbone=backbone,
        pretrained=pretrained,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_gradient_checkpointing=use_gradient_checkpointing,
    ).to(device)
    
    # ===== Wrap with DDP =====
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    gradient_as_bucket_view=False)

    # ===== Optimizer (separate encoder/decoder parameter groups) =====
    actual_model = model.module if hasattr(model, 'module') else model
    encoder_param_ids = {id(p) for p in actual_model.image_encoder.parameters()}
    encoder_params = [p for p in actual_model.parameters() if id(p) in encoder_param_ids]
    decoder_params = [p for p in actual_model.parameters() if id(p) not in encoder_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': weight_decay},
        ]
    )
    
    # ===== Scheduler =====
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    if is_main_process():
        print(f'Steps/epoch: {steps_per_epoch}')
        print(f'Total steps: {total_steps}, Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}%)')
    
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    
    # ===== Resume from checkpoint =====
    start_epoch = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    resume_log_dir = None
    
    if resume_from is not None and os.path.isfile(resume_from):
        if is_main_process():
            print(f'Resuming from checkpoint: {resume_from}')
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        
        # Load model weights
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.load_state_dict(ckpt['model_state_dict'])
        
        # Restore optimizer / scheduler / scaler
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and ckpt.get('scaler_state_dict') is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        # Restore training state — always resume from the next full epoch
        saved_epoch = ckpt.get('epoch', 0)
        start_epoch = saved_epoch + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        resume_log_dir = ckpt.get('log_dir', None)
        
        if is_main_process():
            print(f'  Loaded epoch {saved_epoch}, resuming from epoch {start_epoch}')
            print(f'  best_val_acc={best_val_acc:.4f}, epochs_no_improve={epochs_no_improve}')
        del ckpt
        torch.cuda.empty_cache()
    elif finetune_from is not None and os.path.isfile(finetune_from):
        if is_main_process():
            print(f'Fine-tuning from pretrained weights: {finetune_from}')
        state_dict = torch.load(finetune_from, map_location=device, weights_only=True)
        # Handle both full checkpoint and state_dict-only formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()
        if is_main_process():
            print('  Optimizer/scheduler initialized fresh for fine-tuning')
    
    # Recompute global_step from start_epoch (avoids drift from interrupted runs)
    global_step = (start_epoch - 1) * steps_per_epoch
    
    # TensorBoard (only on rank 0)
    writer = None
    if is_main_process():
        os.makedirs(save_path, exist_ok=True)
        if resume_log_dir is not None and os.path.isdir(resume_log_dir):
            # Reuse existing TensorBoard log directory for continuity
            log_dir = resume_log_dir
            print(f'TensorBoard logs (resumed): {log_dir}')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(save_path, 'logs', timestamp)
            print(f'TensorBoard logs (new): {log_dir}')
        writer = SummaryWriter(log_dir=log_dir)

        # ===== Save hyperparameters =====
        import json
        hparams = {
            'batch_size': batch_size,
            'effective_batch_size': batch_size * world_size,
            'encoder_lr': encoder_lr,
            'decoder_lr': decoder_lr,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'image_size': str(image_size),
            'n_bins': n_bins,
            'backbone': backbone,
            'pretrained': pretrained,
            'd_model': d_model,
            'nhead': nhead,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_atoms': max_atoms,
            'mol_augment': mol_augment,
            'use_amp': use_amp,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'seed': seed,
            'uspto_csv_path': uspto_csv_path,
            'uspto_data_dir': uspto_data_dir,
            'pubchem_smiles_num': pubchem_smiles_num,
            'train_samples': len(train_dataset),
            'resume_from': resume_from,
            'finetune_from': finetune_from,
        }
        hparams_path = os.path.join(log_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
        hparams_md = '| Param | Value |\n|---|---|\n'
        for k, v in hparams.items():
            hparams_md += f'| {k} | {v} |\n'
        writer.add_text('Hyperparameters', hparams_md, global_step=0)
    else:
        log_dir = None
    
    # Broadcast log_dir to all ranks so it can be saved in checkpoint
    if world_size > 1:
        if is_main_process():
            log_dir_bytes = log_dir.encode('utf-8')
            log_dir_len = torch.tensor([len(log_dir_bytes)], dtype=torch.long, device=device)
        else:
            log_dir_len = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(log_dir_len, src=0)
        
        if is_main_process():
            log_dir_tensor = torch.tensor(list(log_dir_bytes), dtype=torch.uint8, device=device)
        else:
            log_dir_tensor = torch.zeros(int(log_dir_len.item()), dtype=torch.uint8, device=device)
        dist.broadcast(log_dir_tensor, src=0)
        
        if not is_main_process():
            log_dir = bytes(log_dir_tensor.cpu().tolist()).decode('utf-8')
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        
        # ===== Critical: update sampler epoch so each rank sees a different shuffle =====
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        num_batches = 0
        
        steps_in_epoch = len(train_loader)
        train_bar = tqdm(total=steps_in_epoch,
                         desc=f'Epoch {epoch} [Train]',
                         disable=not is_main_process())
        train_iter = iter(train_loader)
        
        for _step in range(steps_in_epoch):
            # Fetch batch.  DataLoader timeout=60s catches hung C-extension workers.
            # On any error: kill all workers (recreate loader) and skip this step.
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if is_main_process():
                    warnings.warn(f"DataLoader error at epoch {epoch} step {_step}: {e}")
                batch = None
                # Immediately recreate DataLoader to kill the stuck worker process
                del train_iter
                train_loader = _make_loader()
                train_iter = iter(train_loader)
            
            # ===== DDP-safe batch validity check =====
            # All ranks must agree to skip; otherwise one rank skips the
            # forward/backward while others do the NCCL all-reduce → hang.
            batch_valid = batch is not None
            if world_size > 1:
                valid_tensor = torch.tensor([1 if batch_valid else 0],
                                           dtype=torch.int32, device=device)
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                batch_valid = valid_tensor.item() == 1
            
            if not batch_valid:
                train_bar.update(1)
                continue
            
            images = batch['images'].to(device, non_blocking=True).contiguous()
            tgt_tokens = batch['tgt_tokens'].to(device, non_blocking=True).contiguous()
            tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True).contiguous()
            atom_indices = batch['atom_indices'].to(device, non_blocking=True).contiguous()
            atom_mask = batch['atom_mask'].to(device, non_blocking=True).contiguous()
            max_atoms_val = batch['max_atoms']
            bond_matrices_list = batch['bond_matrices_list']

            # Truncate if needed (safety net; collate_fn already filters > 600)
            tgt_tokens, tgt_padding_mask, atom_indices, atom_mask = truncate_sequences(
                tgt_tokens, tgt_padding_mask, atom_indices, atom_mask, max_seq_len=600
            )
            
            # Teacher forcing
            tgt_input = tgt_tokens[:, :-1].contiguous()
            tgt_target = tgt_tokens[:, 1:].contiguous()
            tgt_input_mask = tgt_padding_mask[:, :-1].contiguous()
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    token_logits, edge_logits, _hidden = model(
                        images=images,
                        tgt_tokens=tgt_input,
                        tgt_key_padding_mask=tgt_input_mask,
                        atom_indices=atom_indices,
                        atom_mask=atom_mask,
                        max_atoms=max_atoms_val
                    )
                    del _hidden  # Free graph branch not needed for loss
                    
                    losses = compute_losses(
                        token_logits=token_logits,
                        edge_logits=edge_logits,
                        tgt_tokens=tgt_target,
                        bond_matrices_list=bond_matrices_list,
                        vocab=vocab,
                        bond_class_weights=bond_class_weights
                    )
                
                scaler.scale(losses['total_loss']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                token_logits, edge_logits, _hidden = model(
                    images=images,
                    tgt_tokens=tgt_input,
                    tgt_key_padding_mask=tgt_input_mask,
                    atom_indices=atom_indices,
                    atom_mask=atom_mask,
                    max_atoms=max_atoms_val
                )
                del _hidden
                
                losses = compute_losses(
                    token_logits=token_logits,
                    edge_logits=edge_logits,
                    tgt_tokens=tgt_target,
                    bond_matrices_list=bond_matrices_list,
                    vocab=vocab,
                    bond_class_weights=bond_class_weights
                )
                
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            running_loss += losses['total_loss'].item()
            num_batches += 1
            global_step += 1
            
            current_lrs = scheduler.get_last_lr()
            
            if is_main_process():
                train_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.2f}",
                    't': f"{losses['token_loss'].item():.2f}",
                    'b': f"{losses['bond_loss'].item():.2f}",
                })
                
                writer.add_scalar('Loss/train', losses['total_loss'].item(), global_step)
                writer.add_scalar('Loss/train_token', losses['token_loss'].item(), global_step)
                writer.add_scalar('Loss/train_bond', losses['bond_loss'].item(), global_step)
                writer.add_scalar('LR/encoder', current_lrs[0], global_step)
                writer.add_scalar('LR/decoder', current_lrs[1], global_step)
            
            train_bar.update(1)
        
        train_bar.close()
        
        # End of epoch summary
        avg_loss = running_loss / max(num_batches, 1)
        if is_main_process():
            print(f'Epoch {epoch} [Train] - avg_loss: {avg_loss:.4f}')
        
        # ===== Evaluation (all ranks participate, metrics aggregated via all_reduce) =====
        val_metrics = validate(
            model=model,
            benchmark_dir=benchmark_dir,
            benchmark_csv_path=benchmark_csv_path,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            beam_size=1,
            max_samples=val_max_samples
        )

        if is_main_process():
            val_acc = val_metrics['exact_match_acc']
            
            # Save best model
            model_to_save = model.module if hasattr(model, 'module') else model
            if val_acc > best_val_acc:
                epochs_no_improve = 0
                best_val_acc = val_acc
                torch.save(model_to_save.state_dict(), os.path.join(save_path, 'best.pth'))
                print(f'New best model saved (exact_match_acc={best_val_acc:.4f}).')
            else:
                epochs_no_improve += 1
                print(f'No improvement for {epochs_no_improve} epoch(s).')
            
            # Save epoch checkpoint (model weights only, for easy standalone loading)
            torch.save(model_to_save.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
            
            # Save full training checkpoint for resume
            checkpoint_state = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'epochs_no_improve': epochs_no_improve,
                'log_dir': log_dir,
            }
            torch.save(checkpoint_state, os.path.join(save_path, 'checkpoint_resume.pth'))
            
            # Broadcast early stopping decision to all ranks
            should_stop = epochs_no_improve >= early_stopping_patience
        else:
            should_stop = False
        
        # ===== Sync early-stopping decision across all ranks =====
        if world_size > 1:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1
        
        if should_stop:
            if is_main_process():
                print(f'Early stopping triggered at epoch {epoch}.')
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(save_path, 'final.pth'))
                writer.close()
            if world_size > 1:
                dist.barrier()
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            cleanup_ddp()
            return []
        
        # Synchronize all ranks before next epoch
        if world_size > 1:
            dist.barrier()
        
        model.train()
    
    # Final save
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(save_path, 'final.pth'))
        writer.close()
        print('Training complete!')

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
    
    cleanup_ddp()
    return []


# ======================== Reward Helpers ========================

def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity: 1 − edit_distance / max(|s1|, |s2|).

    Returns a value in [0, 1] where 1 means identical strings.
    Provides token-level feedback (each correct/incorrect character matters),
    unlike Tanimoto which is insensitive to small edits once fingerprints match.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    # Ensure s1 is the longer string for DP efficiency
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return 1.0 - prev[-1] / max_len


def _fast_render_smiles(smiles: str) -> Tuple[np.ndarray, bool]:
    """Render SMILES to a numpy image using RDKit Draw (~5 ms/call).

    Fast and lightweight, suitable for batch rendering during RL training.
    Returns (H×W×3 uint8, success_bool).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _blank_image(), False
        img_pil = Draw.MolToImage(mol, size=(300, 300))
        img_np = np.array(img_pil)
        # Drop alpha channel if present
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        return img_np, True
    except Exception:
        return _blank_image(), False


# ======================== REINFORCE Loss ========================

def compute_reinforce_loss(
    model: MolScribeModel,
    img_features: torch.Tensor,
    gt_smiles_list: List[str],
    vocab: MolScannerVocab,
    device: torch.device,
    max_len: int = 150,
    baseline: float = 0.0,
    temperature: float = 1.0,
    n_samples: int = 1,
    reward_validity_weight: float = 0.1,
    reward_similarity_weight: float = 0.5,
    reward_exact_match_weight: float = 0.4,
    reward_mode: str = 'tanimoto',
    frozen_encoder: Optional[nn.Module] = None,
    frozen_pos_enc: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict]:
    """REINFORCE loss with composite reward for molecule image recognition.

    Treats the autoregressive decoder as a stochastic policy π_θ.
    The reward is a weighted sum of three signals:

        R = w_v · 𝟙[valid] + w_sim · Similarity(pred, gt) + w_e · 𝟙[exact]

    where Similarity depends on ``reward_mode``:

      - ``'tanimoto'``: Morgan-fingerprint Tanimoto similarity.
      - ``'edit_distance'``: Normalized Levenshtein similarity on canonical
        SMILES (finer-grained, token-level gradient signal).
      - ``'visual'``: Cycle-consistency visual reward — both pred and GT
        SMILES are rendered with RDKit, encoded by a frozen reference
        encoder, and compared via cosine similarity.

    Loss:  L = −(R − baseline) · Σ_t log P(a_t | a_{<t}, x; θ)

    Memory-efficient: sampling uses ``torch.no_grad()``, then a single
    teacher-forced forward pass recomputes log P for backprop.

    When ``n_samples > 1``, self-critical baseline (mean reward across
    samples) is used for variance reduction.

    Args:
        model: MolScribeModel (unwrapped, not DDP).
        img_features: [B, d_model, H, W] encoded image features.
        gt_smiles_list: Ground truth SMILES, length B.
        vocab: MolScannerVocab.
        device: Compute device.
        max_len: Max autoregressive decode length.
        baseline: Constant baseline (overridden by self-critical when n_samples>1).
        temperature: Softmax temperature for exploration.
        n_samples: Samples per image (>1 enables self-critical baseline).
        reward_validity_weight: Weight for valid-SMILES indicator.
        reward_similarity_weight: Weight for the main similarity signal.
        reward_exact_match_weight: Weight for exact-match indicator.
        reward_mode: ``'tanimoto'``, ``'edit_distance'``, or ``'visual'``.
        frozen_encoder: Frozen image_encoder copy for stable visual reward.
        frozen_pos_enc: Frozen pos_enc_2d copy for stable visual reward.

    Returns:
        loss: Scalar REINFORCE loss with gradient graph.
        info: Dict with diagnostic metrics (mean_reward, mean_similarity, etc.).
    """
    B = img_features.size(0)
    total_seqs = B * n_samples

    # Replicate features and GT SMILES for multiple samples
    if n_samples > 1:
        img_features_expanded = img_features.repeat_interleave(n_samples, dim=0)
        gt_expanded = [s for s in gt_smiles_list for _ in range(n_samples)]
    else:
        img_features_expanded = img_features
        gt_expanded = gt_smiles_list

    # ===== Phase 1: Sample sequences with KV-cached decoding (no grad) =====
    # Uses _sample_decode_batch which is O(T) per step instead of O(T²).
    with torch.no_grad():
        sampled_seqs_list = model._sample_decode_batch(
            img_features_expanded, max_len, device, temperature
        )

    # ===== Phase 2: Compute rewards (mode-dependent, non-differentiable) =====
    # Canonicalize GT SMILES: remove atom mapping + functional group handling
    gt_canonical = []
    for s in gt_expanded:
        try:
            s_clean = remove_atom_mapping(s)
            # Also apply functional group replacement + expansion for GT
            s_clean, mappings = _replace_functional_group(s_clean)
            mol_tmp = Chem.MolFromSmiles(s_clean, sanitize=False)
            if mol_tmp is not None:
                s_clean, mol_tmp = _expand_functional_group(mol_tmp, mappings)
            can, ok = canonicalize_smiles(s_clean, ignore_cistrans=True)
            gt_canonical.append(can if ok and can else s)
        except Exception:
            gt_canonical.append(s)

    # Extract predicted SMILES for all samples (with bond prediction for postprocess)
    sampled_smiles = []
    with torch.no_grad():
        for b in range(total_seqs):
            result = vocab.sequence_to_smiles(sampled_seqs_list[b])
            # Run bond predictor to get bond_mat for chirality correction
            try:
                seq_tensor = torch.tensor([sampled_seqs_list[b]], dtype=torch.long, device=device)
                atom_idx, atom_cnt = extract_atom_indices_from_tokens(seq_tensor, vocab)
                a_mask = torch.arange(atom_idx.size(1), device=device) < atom_cnt.unsqueeze(1)
                feat_b = img_features_expanded[b:b+1]
                hidden, _ = model.sequence_decoder(feat_b, seq_tensor)
                e_logits = model.bond_predictor(hidden, atom_idx, a_mask)
                result['bond_mat'] = _symmetrize_edge_predictions(e_logits[0])
            except Exception:
                pass  # bond_mat stays absent; postprocess falls back gracefully
            pred_smiles = _result_to_smiles_postprocess(result)
            if pred_smiles is None:
                pred_smiles = ''
            sampled_smiles.append(pred_smiles)

    # Free Phase 2 intermediates before Phase 3 teacher-forced pass
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Per-sample validity and exact match (shared across all modes)
    is_valid_list = []
    is_exact_list = []
    n_valid_smiles = 0
    n_exact = 0
    for b in range(total_seqs):
        is_valid = sampled_smiles[b] != '' and Chem.MolFromSmiles(sampled_smiles[b]) is not None
        is_exact = sampled_smiles[b] != '' and sampled_smiles[b] == gt_canonical[b]
        is_valid_list.append(is_valid)
        is_exact_list.append(is_exact)
        if is_valid:
            n_valid_smiles += 1
        if is_exact:
            n_exact += 1

    # Compute main similarity signal based on reward_mode
    rewards = torch.zeros(total_seqs, device=device)
    similarity_sum = 0.0
    extra_info: Dict = {}

    if reward_mode == 'tanimoto':
        for b in range(total_seqs):
            tanimoto = compute_tanimoto_similarity(sampled_smiles[b], gt_canonical[b])
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * tanimoto
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += tanimoto
        extra_info['mean_tanimoto'] = similarity_sum / max(total_seqs, 1)

    elif reward_mode == 'edit_distance':
        for b in range(total_seqs):
            edit_sim = _levenshtein_similarity(sampled_smiles[b], gt_canonical[b])
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * edit_sim
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += edit_sim
        extra_info['mean_edit_similarity'] = similarity_sum / max(total_seqs, 1)

    elif reward_mode == 'visual':
        # Both GT and predicted SMILES are rendered with the SAME RDKit
        # renderer to eliminate domain gap. A frozen reference encoder
        # gives stable reward signals that don't drift during training.

        _encode_chunk = 32
        _ref_encoder = frozen_encoder if frozen_encoder is not None else model.image_encoder
        _ref_pos_enc = frozen_pos_enc if frozen_pos_enc is not None else model.pos_enc_2d

        # Render GT SMILES to images with RDKit (same renderer as pred)
        gt_rendered_tensors = []
        for smi in gt_canonical:
            img_np, _ = _fast_render_smiles(smi)
            gt_rendered_tensors.append(
                model.inference_transforms(image=img_np)['image']
            )
        gt_rendered_batch = torch.stack(gt_rendered_tensors).to(device)

        # Encode GT rendered images in chunks (no grad, using frozen encoder)
        gt_feat_chunks = []
        with torch.no_grad():
            for i in range(0, total_seqs, _encode_chunk):
                chunk = gt_rendered_batch[i:i + _encode_chunk]
                feat = _ref_encoder(chunk)
                feat = _ref_pos_enc(feat)
                feat = feat.float().mean(dim=[2, 3])
                gt_feat_chunks.append(feat)
        gt_feat = torch.cat(gt_feat_chunks, dim=0)  # [total_seqs, d_model]
        del gt_rendered_batch

        # Render all predicted SMILES to images with RDKit
        pred_rendered_tensors = []
        render_success_count = 0
        for smi in sampled_smiles:
            img_np, ok = _fast_render_smiles(smi)
            pred_rendered_tensors.append(
                model.inference_transforms(image=img_np)['image']
            )
            if ok:
                render_success_count += 1
        pred_rendered_batch = torch.stack(pred_rendered_tensors).to(device)

        # Encode predicted rendered images in chunks (no grad, using frozen encoder)
        pred_feat_chunks = []
        with torch.no_grad():
            for i in range(0, total_seqs, _encode_chunk):
                chunk = pred_rendered_batch[i:i + _encode_chunk]
                feat = _ref_encoder(chunk)
                feat = _ref_pos_enc(feat)
                feat = feat.float().mean(dim=[2, 3])
                pred_feat_chunks.append(feat)
        pred_feat = torch.cat(pred_feat_chunks, dim=0)  # [total_seqs, d_model]
        del pred_rendered_batch

        # Cosine similarity, clamped to [0, 1]
        cos_sims = F.cosine_similarity(gt_feat, pred_feat, dim=1).clamp(min=0.0)

        for b in range(total_seqs):
            visual_sim = cos_sims[b].item()
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * visual_sim
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += visual_sim

        extra_info['mean_visual_similarity'] = similarity_sum / max(total_seqs, 1)
        extra_info['render_success'] = render_success_count
        del pred_feat, gt_feat

    else:
        raise ValueError(f"Unknown reward_mode='{reward_mode}'. "
                         f"Choose from 'tanimoto', 'edit_distance', 'visual'.")

    n_valid = sum(1 for b in range(total_seqs)
                  if rewards[b].item() > reward_validity_weight + 1e-6)

    # ===== Phase 3: One teacher-forced forward pass to get log P(sampled) =====
    # Pad sampled sequences to the same length
    max_seq_len = max(len(s) for s in sampled_seqs_list)
    padded_seqs = torch.full((total_seqs, max_seq_len), vocab.pad_idx,
                             dtype=torch.long, device=device)
    seq_lengths = torch.zeros(total_seqs, dtype=torch.long, device=device)
    for b, seq in enumerate(sampled_seqs_list):
        padded_seqs[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        seq_lengths[b] = len(seq)

    # Teacher-forced forward: input = seq[:-1], target = seq[1:]
    tf_input = padded_seqs[:, :-1]
    tf_target = padded_seqs[:, 1:]
    T_out = tf_target.size(1)

    # This single forward pass IS differentiable w.r.t. decoder parameters
    _hidden, tf_logits = model.sequence_decoder(img_features_expanded, tf_input)
    # tf_logits: [total_seqs, T_out, vocab_size]

    # Compute per-token log P
    log_probs = F.log_softmax(tf_logits / temperature, dim=-1)  # [total_seqs, T_out, V]
    # Gather log P of the actually-sampled tokens
    token_log_probs = log_probs.gather(2, tf_target.unsqueeze(-1)).squeeze(-1)  # [total_seqs, T_out]

    # Mask out padding positions
    pad_mask = (tf_target != vocab.pad_idx).float()  # [total_seqs, T_out]
    log_probs_sum = (token_log_probs * pad_mask).sum(dim=1)  # [total_seqs]

    # ===== Phase 4: REINFORCE loss =====
    if n_samples > 1:
        rewards_reshaped = rewards.view(B, n_samples)
        sc_baseline = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages = (rewards_reshaped - sc_baseline).view(-1).detach()
    else:
        advantages = (rewards - baseline).detach()

    reinforce_loss = -(advantages * log_probs_sum).mean()

    info = {
        'mean_reward': rewards.mean().item(),
        'mean_log_prob': log_probs_sum.mean().item(),
        'sampled_smiles': sampled_smiles[:B],
        'n_valid': n_valid,
        'mean_similarity': similarity_sum / max(total_seqs, 1),
        'exact_matches': n_exact,
        'valid_smiles_count': n_valid_smiles,
        'reward_mode': reward_mode,
    }
    info.update(extra_info)  # mode-specific keys (mean_tanimoto, mean_edit_similarity, etc.)

    return reinforce_loss, info


# ======================== Minimum Risk Training (MRT) Loss ========================

def compute_mrt_loss(
    model: MolScribeModel,
    img_features: torch.Tensor,
    gt_smiles_list: List[str],
    vocab: MolScannerVocab,
    device: torch.device,
    max_len: int = 150,
    temperature: float = 1.0,
    n_samples: int = 8,
    mrt_alpha: float = 1.0,
    reward_validity_weight: float = 0.1,
    reward_similarity_weight: float = 0.5,
    reward_exact_match_weight: float = 0.4,
    reward_mode: str = 'tanimoto',
    frozen_encoder: Optional[nn.Module] = None,
    frozen_pos_enc: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Minimum Risk Training (MRT) loss for molecule image recognition.

    Minimizes expected cost (1 − reward) under the model's sampling
    distribution, approximated by N candidates per image.

    Per-image loss:
        L_MRT = Σ_i  w_i · (1 − R_i)
        w_i = softmax(α · log p(y_i | x))_i

    Key advantages over REINFORCE:
      - Pathwise gradients through softmax weights → per-token signal.
      - Bounded gradients from softmax normalization → lower variance.
      - No baseline needed.

    Args:
        model: MolScribeModel (unwrapped, not DDP).
        img_features: [B, d_model, H, W] with computation graph.
        gt_smiles_list: Ground truth SMILES, length B.
        vocab: MolScannerVocab.
        device: Compute device.
        max_len: Max autoregressive decode length.
        temperature: Sampling temperature for candidates.
        n_samples: Candidates per image (≥ 2).
        mrt_alpha: Softmax sharpness over candidate log-probs.
        reward_validity_weight: Weight for valid-SMILES indicator.
        reward_similarity_weight: Weight for main similarity signal.
        reward_exact_match_weight: Weight for exact-match indicator.
        reward_mode: ``'tanimoto'``, ``'edit_distance'``, or ``'visual'``.
        frozen_encoder: Frozen encoder for stable visual reward.
        frozen_pos_enc: Frozen pos_enc_2d for stable visual reward.

    Returns:
        loss: Scalar MRT loss with gradient graph.
        info: Dict with diagnostic metrics.
    """
    B = img_features.size(0)
    total_seqs = B * n_samples

    # Replicate features and GT SMILES for N samples per image
    img_features_expanded = img_features.repeat_interleave(n_samples, dim=0)
    gt_expanded = [s for s in gt_smiles_list for _ in range(n_samples)]

    # ===== Phase 1: Sample N candidate sequences (no grad) =====
    with torch.no_grad():
        sampled_seqs_list = model._sample_decode_batch(
            img_features_expanded, max_len, device, temperature
        )

    # ===== Phase 2: Compute rewards → costs (non-differentiable) =====
    gt_canonical = []
    for s in gt_expanded:
        try:
            s_clean = remove_atom_mapping(s)
            s_clean, mappings = _replace_functional_group(s_clean)
            mol_tmp = Chem.MolFromSmiles(s_clean, sanitize=False)
            if mol_tmp is not None:
                s_clean, mol_tmp = _expand_functional_group(mol_tmp, mappings)
            can, ok = canonicalize_smiles(s_clean, ignore_cistrans=True)
            gt_canonical.append(can if ok and can else s)
        except Exception:
            gt_canonical.append(s)

    # Extract predicted SMILES (with bond prediction + postprocess)
    sampled_smiles = []
    with torch.no_grad():
        for b in range(total_seqs):
            result = vocab.sequence_to_smiles(sampled_seqs_list[b])
            try:
                seq_tensor = torch.tensor([sampled_seqs_list[b]], dtype=torch.long, device=device)
                atom_idx, atom_cnt = extract_atom_indices_from_tokens(seq_tensor, vocab)
                a_mask = torch.arange(atom_idx.size(1), device=device) < atom_cnt.unsqueeze(1)
                feat_b = img_features_expanded[b:b+1]
                hidden, _ = model.sequence_decoder(feat_b, seq_tensor)
                e_logits = model.bond_predictor(hidden, atom_idx, a_mask)
                result['bond_mat'] = _symmetrize_edge_predictions(e_logits[0])
            except Exception:
                pass
            pred_smiles = _result_to_smiles_postprocess(result)
            if pred_smiles is None:
                pred_smiles = ''
            sampled_smiles.append(pred_smiles)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Per-sample validity and exact match
    is_valid_list = []
    is_exact_list = []
    n_valid_smiles = 0
    n_exact = 0
    for b in range(total_seqs):
        is_valid = sampled_smiles[b] != '' and Chem.MolFromSmiles(sampled_smiles[b]) is not None
        is_exact = sampled_smiles[b] != '' and sampled_smiles[b] == gt_canonical[b]
        is_valid_list.append(is_valid)
        is_exact_list.append(is_exact)
        if is_valid:
            n_valid_smiles += 1
        if is_exact:
            n_exact += 1

    # Compute rewards based on reward_mode
    rewards = torch.zeros(total_seqs, device=device)
    similarity_sum = 0.0
    extra_info: Dict = {}

    if reward_mode == 'tanimoto':
        for b in range(total_seqs):
            tanimoto = compute_tanimoto_similarity(sampled_smiles[b], gt_canonical[b])
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * tanimoto
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += tanimoto
        extra_info['mean_tanimoto'] = similarity_sum / max(total_seqs, 1)

    elif reward_mode == 'edit_distance':
        for b in range(total_seqs):
            edit_sim = _levenshtein_similarity(sampled_smiles[b], gt_canonical[b])
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * edit_sim
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += edit_sim
        extra_info['mean_edit_similarity'] = similarity_sum / max(total_seqs, 1)

    elif reward_mode == 'visual':
        _encode_chunk = 32
        _ref_encoder = frozen_encoder if frozen_encoder is not None else model.image_encoder
        _ref_pos_enc = frozen_pos_enc if frozen_pos_enc is not None else model.pos_enc_2d

        gt_rendered_tensors = []
        for smi in gt_canonical:
            img_np, _ = _fast_render_smiles(smi)
            gt_rendered_tensors.append(model.inference_transforms(image=img_np)['image'])
        gt_rendered_batch = torch.stack(gt_rendered_tensors).to(device)

        gt_feat_chunks = []
        with torch.no_grad():
            for i in range(0, total_seqs, _encode_chunk):
                chunk = gt_rendered_batch[i:i + _encode_chunk]
                feat = _ref_encoder(chunk)
                feat = _ref_pos_enc(feat)
                feat = feat.float().mean(dim=[2, 3])
                gt_feat_chunks.append(feat)
        gt_feat = torch.cat(gt_feat_chunks, dim=0)
        del gt_rendered_batch

        pred_rendered_tensors = []
        render_success_count = 0
        for smi in sampled_smiles:
            img_np, ok = _fast_render_smiles(smi)
            pred_rendered_tensors.append(model.inference_transforms(image=img_np)['image'])
            if ok:
                render_success_count += 1
        pred_rendered_batch = torch.stack(pred_rendered_tensors).to(device)

        pred_feat_chunks = []
        with torch.no_grad():
            for i in range(0, total_seqs, _encode_chunk):
                chunk = pred_rendered_batch[i:i + _encode_chunk]
                feat = _ref_encoder(chunk)
                feat = _ref_pos_enc(feat)
                feat = feat.float().mean(dim=[2, 3])
                pred_feat_chunks.append(feat)
        pred_feat = torch.cat(pred_feat_chunks, dim=0)
        del pred_rendered_batch

        cos_sims = F.cosine_similarity(gt_feat, pred_feat, dim=1).clamp(min=0.0)
        for b in range(total_seqs):
            visual_sim = cos_sims[b].item()
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * visual_sim
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += visual_sim
        extra_info['mean_visual_similarity'] = similarity_sum / max(total_seqs, 1)
        extra_info['render_success'] = render_success_count
        del pred_feat, gt_feat

    else:
        raise ValueError(f"Unknown reward_mode='{reward_mode}'.")

    # Costs = 1 - reward (detached, no grad)
    costs = (1.0 - rewards).detach()  # [B * N]

    # ===== Phase 3: Teacher-forced forward pass → log p(y_i | x) =====
    max_seq_len = max(len(s) for s in sampled_seqs_list)
    padded_seqs = torch.full((total_seqs, max_seq_len), vocab.pad_idx,
                             dtype=torch.long, device=device)
    for b, seq in enumerate(sampled_seqs_list):
        padded_seqs[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    tf_input = padded_seqs[:, :-1]
    tf_target = padded_seqs[:, 1:]

    # Differentiable forward pass
    _hidden, tf_logits = model.sequence_decoder(img_features_expanded, tf_input)

    # Per-token log probabilities
    log_probs = F.log_softmax(tf_logits, dim=-1)
    token_log_probs = log_probs.gather(2, tf_target.unsqueeze(-1)).squeeze(-1)

    pad_mask = (tf_target != vocab.pad_idx).float()
    seq_log_probs = (token_log_probs * pad_mask).sum(dim=1)  # [B * N]

    # ===== Phase 4: MRT loss =====
    # Group by image: [B, N]
    seq_log_probs_grouped = seq_log_probs.view(B, n_samples)
    costs_grouped = costs.view(B, n_samples)

    # Softmax-normalized weights over candidates per image
    # Gradients flow through seq_log_probs → decoder → encoder
    weights = F.softmax(mrt_alpha * seq_log_probs_grouped, dim=1)  # [B, N]

    # Weighted average cost per image, then mean over batch
    mrt_loss = (weights * costs_grouped).sum(dim=1).mean()

    # Diagnostics
    n_valid = sum(1 for b in range(total_seqs)
                  if rewards[b].item() > reward_validity_weight + 1e-6)

    info = {
        'mean_reward': rewards.mean().item(),
        'mean_cost': costs.mean().item(),
        'mean_log_prob': seq_log_probs.mean().item(),
        'sampled_smiles': sampled_smiles[:B],
        'n_valid': n_valid,
        'mean_similarity': similarity_sum / max(total_seqs, 1),
        'exact_matches': n_exact,
        'valid_smiles_count': n_valid_smiles,
        'reward_mode': reward_mode,
    }
    info.update(extra_info)

    return mrt_loss, info

# ======================== RL Finetuning on Real Images ========================

class RealImageDataset(Dataset):
    """Dataset that loads real-world molecule images + GT SMILES from CSV.

    CSV columns: image_id, file_path, SMILES
    Returns (image_tensor, gt_smiles_str) pairs.
    """

    def __init__(self, csv_path: str, image_dir: str,
                 transforms, image_size: Tuple[int, int] = (384, 384),
                 max_samples: Optional[int] = None):
        label_df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples < len(label_df):
            label_df = label_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.items = []
        for _, row in label_df.iterrows():
            img_id = str(row['image_id'])
            img_path = os.path.join(image_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                continue
            gt_smi = str(row['SMILES'])
            self.items.append({'img_path': img_path, 'gt_smiles': gt_smi})

        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item['img_path']).convert('RGB')
            tensor = self.transforms(image=np.array(img))['image']
        except Exception:
            tensor = torch.zeros(3, *self.image_size)
        return tensor, item['gt_smiles']


def _real_collate_fn(batch):
    """Simple collate: stack images, collect SMILES strings."""
    images = torch.stack([b[0] for b in batch])
    smiles = [b[1] for b in batch]
    return images, smiles


def train_rl_real_finetune(
    # ===== Data =====
    train_csv_paths: List[str],
    train_image_dirs: List[str],
    # ===== Validation =====
    val_benchmark_dir: str = '',
    val_benchmark_csv: str = '',
    val_max_samples: Optional[int] = None,
    # ===== Model =====
    pretrained_path: str = '',
    save_path: str = '',
    # ===== Architecture (must match pretrained) =====
    image_size: Tuple[int, int] = (384, 384),
    n_bins: int = 64,
    backbone: str = 'swin_b',
    d_model: int = 256,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    # ===== Training =====
    num_epochs: int = 10,
    batch_size: int = 16,
    encoder_lr: float = 1e-5,
    decoder_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    warmup_ratio: float = 0.02,
    seed: int = 2026,
    early_stopping_patience: int = 10,
    num_workers: int = 4,
    use_amp: bool = True,
    force_cpu: bool = False,
    # ===== RL =====
    rl_max_len: int = 500,
    rl_temperature: float = 0.8,
    rl_n_samples: int = 16,
    rl_subsample: int = 16,
    reward_validity_weight: float = 0.1,
    reward_similarity_weight: float = 0.5,
    reward_exact_match_weight: float = 0.4,
    reward_mode: str = 'visual',
    # ===== RL method =====
    rl_method: str = 'reinforce',
    mrt_alpha: float = 1.0,
    # ===== Resume =====
    resume_from: Optional[str] = None,
) -> List[float]:
    """Pure RL finetuning on real-world molecule images (no MLE loss).

    Finetunes a pretrained MolScribe model using only sequence-level RL
    loss (REINFORCE or MRT) with a composite reward:

        R = w_v · 𝟙[valid] + w_sim · Similarity + w_e · 𝟙[exact match]

    Reward modes (``reward_mode``):
      - ``'visual'``: cycle-consistency — render pred & GT SMILES with
        RDKit, encode with frozen reference encoder, compare via cosine
        similarity. Works without GT SMILES labels (beyond exact match).
      - ``'tanimoto'``: Morgan-fingerprint Tanimoto similarity.
      - ``'edit_distance'``: normalized Levenshtein similarity.

    RL methods (``rl_method``):
      - ``'reinforce'``: REINFORCE with self-critical baseline.
      - ``'mrt'``: Minimum Risk Training (lower variance, no baseline).

    A frozen copy of the pretrained encoder provides stable visual reward
    signals throughout training.

    Launch:
        torchrun --nproc_per_node=<N> MolScanner/train_rl_real.py
    """
    # ===== DDP setup =====
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        setup_ddp(local_rank, world_size)

    if force_cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    if is_main_process():
        print(f'=== Real-Image RL Finetuning (cycle consistency) ===')
        print(f'Device: {device}, World size: {world_size}')
        print(f'Train CSVs: {train_csv_paths}')
        print(f'rl_temperature={rl_temperature}, rl_n_samples={rl_n_samples}, '
              f'rl_subsample={rl_subsample}')
        print(f'reward weights: validity={reward_validity_weight}, '
              f'similarity={reward_similarity_weight}, '
              f'exact_match={reward_exact_match_weight}')
        print(f'rl_method={rl_method}, mrt_alpha={mrt_alpha}, '
              f'reward_mode={reward_mode}')

    # ===== Vocab + Model =====
    vocab = MolScannerVocab(n_bins=n_bins)

    model = MolScribeModel(
        vocab=vocab,
        backbone=backbone,
        pretrained=False,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        image_size=image_size,
    ).to(device)

    if is_main_process():
        print(f'Loading pretrained weights from {pretrained_path} ...')
    model.load_model(pretrained_path, device=device)

    # Frozen reference encoder for stable visual reward
    import copy
    frozen_encoder = copy.deepcopy(model.image_encoder).eval()
    frozen_pos_enc = copy.deepcopy(model.pos_enc_2d).eval()
    frozen_encoder.requires_grad_(False)
    frozen_pos_enc.requires_grad_(False)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
                    gradient_as_bucket_view=True)

    actual_model = model.module if hasattr(model, 'module') else model

    # ===== Dataset =====
    all_items = []
    for csv_path, img_dir in zip(train_csv_paths, train_image_dirs):
        ds = RealImageDataset(csv_path, img_dir, actual_model.inference_transforms,
                              image_size)
        all_items.extend(ds.items)
        if is_main_process():
            print(f'  Loaded {len(ds)} images from {csv_path}')

    # Build a single dataset from merged items
    train_dataset = RealImageDataset.__new__(RealImageDataset)
    train_dataset.items = all_items
    train_dataset.transforms = actual_model.inference_transforms
    train_dataset.image_size = image_size

    if is_main_process():
        print(f'Total training images: {len(train_dataset)}')

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=local_rank,
        shuffle=True, seed=seed,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=_real_collate_fn,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # ===== Optimizer =====
    encoder_param_ids = {id(p) for p in actual_model.image_encoder.parameters()}
    encoder_params = [p for p in actual_model.parameters() if id(p) in encoder_param_ids]
    decoder_params = [p for p in actual_model.parameters() if id(p) not in encoder_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': weight_decay},
    ])

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None

    # ===== Resume =====
    start_epoch = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    rl_baseline_ema = 0.0
    resume_log_dir = None

    if resume_from is not None and os.path.isfile(resume_from):
        if is_main_process():
            print(f'Resuming from {resume_from}')
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        actual_model.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and ckpt.get('scaler_state_dict') is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        rl_baseline_ema = ckpt.get('rl_baseline_ema', 0.0)
        resume_log_dir = ckpt.get('log_dir', None)
        if is_main_process():
            print(f'  Resuming from epoch {start_epoch}, '
                  f'best_val_acc={best_val_acc:.4f}')
        del ckpt
        torch.cuda.empty_cache()

    global_step = (start_epoch - 1) * steps_per_epoch

    # ===== TensorBoard =====
    writer = None
    log_dir = None
    if is_main_process():
        os.makedirs(save_path, exist_ok=True)
        if resume_log_dir and os.path.isdir(resume_log_dir):
            log_dir = resume_log_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(save_path, 'logs', f'real_rl_{timestamp}')
        writer = SummaryWriter(log_dir=log_dir)
        print(f'TensorBoard logs: {log_dir}')

        # ===== Save hyperparameters =====
        import json
        hparams = {
            'batch_size': batch_size,
            'effective_batch_size': batch_size * world_size,
            'encoder_lr': encoder_lr,
            'decoder_lr': decoder_lr,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'num_epochs': num_epochs,
            'image_size': str(image_size),
            'n_bins': n_bins,
            'backbone': backbone,
            'd_model': d_model,
            'nhead': nhead,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'rl_method': rl_method,
            'rl_max_len': rl_max_len,
            'rl_temperature': rl_temperature,
            'rl_n_samples': rl_n_samples,
            'rl_subsample': rl_subsample,
            'reward_mode': reward_mode,
            'reward_validity_weight': reward_validity_weight,
            'reward_similarity_weight': reward_similarity_weight,
            'reward_exact_match_weight': reward_exact_match_weight,
            'mrt_alpha': mrt_alpha if rl_method == 'mrt' else None,
            'seed': seed,
            'pretrained_path': pretrained_path,
            'train_csv_paths': train_csv_paths,
            'train_images_total': len(train_dataset),
        }
        hparams_path = os.path.join(log_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
        hparams_md = '| Param | Value |\n|---|---|\n'
        for k, v in hparams.items():
            hparams_md += f'| {k} | {v} |\n'
        writer.add_text('Hyperparameters', hparams_md, global_step=0)

    # Broadcast log_dir to all ranks
    if world_size > 1:
        if is_main_process():
            log_dir_bytes = log_dir.encode('utf-8')
            log_dir_len = torch.tensor([len(log_dir_bytes)], dtype=torch.long, device=device)
        else:
            log_dir_len = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(log_dir_len, src=0)
        if not is_main_process():
            log_dir_bytes = bytes(log_dir_len.item())
        log_dir_tensor = torch.tensor(list(log_dir_bytes if is_main_process()
                                           else b'\x00' * log_dir_len.item()),
                                      dtype=torch.uint8, device=device)
        dist.broadcast(log_dir_tensor, src=0)
        if not is_main_process():
            log_dir = bytes(log_dir_tensor.cpu().tolist()).decode('utf-8')

    if is_main_process():
        print(f'\nSteps/epoch: {steps_per_epoch}, Total steps: {total_steps}, '
              f'Warmup steps: {warmup_steps}')
        print(f'Starting from epoch {start_epoch}\n')

    # ===== Training Loop =====
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_reward = 0.0
        running_loss = 0.0
        num_batches = 0

        pbar = (tqdm(train_loader, desc=f'Epoch {epoch}', dynamic_ncols=True)
                if is_main_process() else train_loader)

        for images, gt_smiles_batch in pbar:
            images = images.to(device, non_blocking=True)
            rl_subsample_actual = min(rl_subsample, images.size(0))

            optimizer.zero_grad(set_to_none=True)

            # Encode images
            rl_images = images[:rl_subsample_actual]
            rl_gt = gt_smiles_batch[:rl_subsample_actual]

            img_features = actual_model.image_encoder(rl_images)
            img_features = actual_model.pos_enc_2d(img_features)
            if use_amp:
                img_features = img_features.float()

            if rl_method == 'mrt':
                rl_loss, rl_info = compute_mrt_loss(
                    model=actual_model,
                    img_features=img_features,
                    gt_smiles_list=rl_gt,
                    vocab=vocab,
                    device=device,
                    max_len=rl_max_len,
                    temperature=rl_temperature,
                    n_samples=rl_n_samples,
                    mrt_alpha=mrt_alpha,
                    reward_validity_weight=reward_validity_weight,
                    reward_similarity_weight=reward_similarity_weight,
                    reward_exact_match_weight=reward_exact_match_weight,
                    reward_mode=reward_mode,
                    frozen_encoder=frozen_encoder,
                    frozen_pos_enc=frozen_pos_enc,
                )
            else:
                rl_loss, rl_info = compute_reinforce_loss(
                    model=actual_model,
                    img_features=img_features,
                    gt_smiles_list=rl_gt,
                    vocab=vocab,
                    device=device,
                    max_len=rl_max_len,
                    baseline=rl_baseline_ema,
                    temperature=rl_temperature,
                    n_samples=rl_n_samples,
                    reward_validity_weight=reward_validity_weight,
                    reward_similarity_weight=reward_similarity_weight,
                    reward_exact_match_weight=reward_exact_match_weight,
                    reward_mode=reward_mode,
                    frozen_encoder=frozen_encoder,
                    frozen_pos_enc=frozen_pos_enc,
                )

            rl_reward = rl_info['mean_reward']

            if scaler is not None:
                scaler.scale(rl_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            rl_baseline_ema = 0.9 * rl_baseline_ema + 0.1 * rl_reward

            running_reward += rl_reward
            running_loss += rl_loss.item()
            num_batches += 1
            global_step += 1

            if is_main_process():
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'reward': f"{rl_reward:.3f}",
                        'loss': f"{rl_loss.item():.3f}",
                        'valid': rl_info.get('valid_smiles_count', 0),
                        'exact': rl_info.get('exact_matches', 0),
                    })

                if writer:
                    writer.add_scalar('RL/reward_composite', rl_reward, global_step)
                    writer.add_scalar('RL/similarity',
                                      rl_info.get('mean_similarity', 0.0), global_step)
                    writer.add_scalar('RL/valid_smiles',
                                      rl_info.get('valid_smiles_count', 0), global_step)
                    writer.add_scalar('RL/exact_matches',
                                      rl_info.get('exact_matches', 0), global_step)
                    writer.add_scalar('RL/baseline_ema', rl_baseline_ema, global_step)
                    writer.add_scalar('RL/loss', rl_loss.item(), global_step)
                    writer.add_scalar('RL/render_success',
                                      rl_info.get('render_success', 0), global_step)
                    writer.add_scalar('LR/encoder',
                                      optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('LR/decoder',
                                      optimizer.param_groups[1]['lr'], global_step)

        # ===== Epoch summary =====
        avg_reward = running_reward / max(num_batches, 1)
        avg_loss = running_loss / max(num_batches, 1)
        if is_main_process():
            print(f'\nEpoch {epoch}: avg_reward={avg_reward:.4f}, avg_loss={avg_loss:.4f}')

        # ===== Validation =====
        if val_benchmark_dir and val_benchmark_csv:
            val_results = validate(
                model=model,
                benchmark_dir=val_benchmark_dir,
                benchmark_csv_path=val_benchmark_csv,
                device=device,
                epoch=epoch,
                writer=writer,
                global_step=global_step,
                beam_size=1,
                max_samples=val_max_samples,
            )

            val_acc = val_results.get('exact_match_acc', 0.0)

            if is_main_process():
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    torch.save(actual_model.state_dict(),
                               os.path.join(save_path, 'best.pth'))
                    print(f'  New best: {best_val_acc:.4f}')
                else:
                    epochs_no_improve += 1
                    print(f'  No improvement ({epochs_no_improve}/{early_stopping_patience})')
        else:
            val_acc = 0.0

        # ===== Checkpoint =====
        if is_main_process():
            ckpt = {
                'epoch': epoch,
                'model_state_dict': actual_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_acc': best_val_acc,
                'epochs_no_improve': epochs_no_improve,
                'rl_baseline_ema': rl_baseline_ema,
                'log_dir': log_dir,
            }
            torch.save(ckpt, os.path.join(save_path, 'checkpoint_resume.pth'))
            torch.save(actual_model.state_dict(),
                       os.path.join(save_path, f'epoch_{epoch}.pth'))
            del ckpt

        # ===== Early stopping =====
        if epochs_no_improve >= early_stopping_patience and early_stopping_patience > 0:
            if is_main_process():
                print(f'Early stopping at epoch {epoch}')
            break

        if world_size > 1:
            dist.barrier()
        model.train()

    # ===== Final save =====
    if is_main_process():
        torch.save(actual_model.state_dict(), os.path.join(save_path, 'final.pth'))
        if writer:
            writer.close()
        print('Real-image RL finetuning complete!')

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
    cleanup_ddp()
    return []
