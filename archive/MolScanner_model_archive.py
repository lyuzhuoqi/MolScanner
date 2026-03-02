import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from func_timeout import FunctionTimedOut
import math
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# ======================== Performance Optimizations ========================
# Disable cuDNN benchmark when using DataParallel with variable inputs to avoid memory issues
# Can be re-enabled for single-GPU with fixed input sizes
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Use TensorFloat-32 for faster matmul operations on Ampere+ GPUs (no accuracy loss for training)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
# Disable cv2 threading globally to avoid conflicts with DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from drawing_engine import generate_image_from_smiles
from chemistry import _convert_graph_to_smiles, canonicalize_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# ======================== Vocabulary & Tokenizer ========================

class MolScannerVocab:
    """Character-level vocabulary for atom sequences with separate X/Y coordinate bins."""
    
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
    
    def tokenize_atom_seq(self, atom_seq: List) -> List[int]:
        """
        Tokenize atom sequence to token indices.
        Structure: [SOS, SymbolChars..., X_BIN, Y_BIN, SymbolChars..., X_BIN, Y_BIN, ..., EOS]
        """
        tokens = [self.sos_idx]
        
        i = 0
        while i < len(atom_seq):
            symbol = atom_seq[i]
            x_bin = atom_seq[i + 1]
            y_bin = atom_seq[i + 2]
            
            # Tokenize symbol first (Directly append chars)
            for char in symbol:
                tokens.append(self.token2idx.get(char, self.unk_idx))
            
            # Tokenize coordinates using separate X and Y bin tokens
            tokens.append(self.token2idx[f'<X_BIN_{x_bin}>'])
            tokens.append(self.token2idx[f'<Y_BIN_{y_bin}>'])
            
            i += 3
        
        tokens.append(self.eos_idx)
        return tokens
    
    def detokenize_atom_seq(self, token_indices: List[int]) -> Tuple[List, bool]:
        """
        Detokenize token indices back to atom sequence with strict validation.
        Structure: [Char, Char, ...] -> [X_BIN] -> [Y_BIN] -> Repeat
        """
        atom_seq = []
        i = 0
        
        # 1. Skip SOS
        if len(token_indices) > 0 and token_indices[0] == self.sos_idx:
            i = 1
        
        try:
            while i < len(token_indices):
                token = token_indices[i]
                
                # --- Case A: End of Sequence ---
                if token == self.eos_idx:
                    break
                
                # --- Case B: Symbol Character (Start of atom) ---
                is_coord = self.is_x_coord_token(token) or self.is_y_coord_token(token)
                is_special = token in [self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx]
                
                if not is_coord and not is_special:
                    # 读取符号字符直到遇到 X_BIN
                    symbol_chars = []
                    while i < len(token_indices):
                        current_token = token_indices[i]
                        if self.is_x_coord_token(current_token):
                            break
                        if current_token == self.eos_idx:
                            break
                        if self.is_y_coord_token(current_token):
                            # Y_BIN 不应该在 X_BIN 之前出现
                            return atom_seq, False
                        char = self.idx2token.get(current_token, '?')
                        symbol_chars.append(char)
                        i += 1
                    
                    # 校验: 必须有符号字符
                    if not symbol_chars:
                        return atom_seq, False
                    
                    # 校验: 符号后面必须紧跟 X 坐标 (除非是 EOS)
                    if i >= len(token_indices):
                        return atom_seq, False
                    
                    x_token_idx = token_indices[i]
                    if x_token_idx == self.eos_idx:
                        # 符号后面直接遇到 EOS，不完整的原子
                        return atom_seq, False
                    if not self.is_x_coord_token(x_token_idx):
                        return atom_seq, False  # 符号后面必须跟 X_BIN
                    
                    # 获取 X 坐标 - 解析 <X_BIN_123>
                    x_bin_token = self.idx2token[x_token_idx]
                    x_val = int(x_bin_token[7:-1])  # Skip '<X_BIN_' and '>'
                    
                    # 校验: X 后面必须紧跟 Y 坐标
                    if i + 1 >= len(token_indices):
                        return atom_seq, False
                    
                    y_token_idx = token_indices[i + 1]
                    if not self.is_y_coord_token(y_token_idx):
                        return atom_seq, False  # X 后面必须跟 Y_BIN
                    
                    # 获取 Y 坐标 - 解析 <Y_BIN_123>
                    y_bin_token = self.idx2token[y_token_idx]
                    y_val = int(y_bin_token[7:-1])  # Skip '<Y_BIN_' and '>'
                    
                    i += 2  # 跳过 X 和 Y
                    
                    # --- Commit Atom ---
                    symbol = "".join(symbol_chars)
                    atom_seq.extend([symbol, x_val, y_val])
                
                # --- Case C: Unexpected Coordinate Token at start ---
                elif is_coord:
                    # 坐标 token 不应该在符号之前出现
                    return atom_seq, False
                
                # --- Case D: Unexpected special token ---
                else:
                    return atom_seq, False
            
            # 如果序列为空且没有正常解析出任何原子
            if len(atom_seq) == 0:
                return atom_seq, False 
            
            return atom_seq, True

        except Exception:
            return atom_seq, False

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
    """Dataset for generating molecule images, atom sequences, and bond matrices."""
    
    def __init__(
        self, 
        vocab,
        smiles_list: List[str],
        shuffle_smiles: bool = True,
        image_size: Tuple[int, int] = (384, 384),
        mol_augment: bool = True,
        img_augment: bool = True,
        geo_augment: bool = True,
        atom_shuffle: bool = True,
        default_drawing_style: bool = False,
    ):
        self.vocab = vocab
        self.smiles_list = smiles_list
        
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
        self.atom_shuffle = atom_shuffle
        self.default_drawing_style = default_drawing_style
        
        # Separate pad to square operation (handles keypoints manually)
        self.pad_to_square = PadToSquare(fill=255)

        # 1. Rotation with fit_output (Albumentations handles keypoints)
        self.geo_transforms_list = []
        if self.geo_augment:
            self.geo_transforms_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=1.0),
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
        self.img_transforms = A.Compose(self.img_transforms_list, 
                                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        # 3. Final transforms (after padding, no keypoints needed)
        self.post_transforms = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_AREA),
            A.ToGray(num_output_channels=3),  # Keep 3 channels for pretrained backbone
            A.Normalize(),  # ImageNet normalization
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        anchor_smiles = self.smiles_list[idx]

        # Initialize defaults to handle exceptions safely
        img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.uint8)
        success = False
        graph = None

        try:
            img, _, graph, success, _, _ = generate_image_from_smiles(
                anchor_smiles, n_bins=None, 
                mol_augment=self.mol_augment, 
                default_drawing_style=self.default_drawing_style,
                shuffle_nodes=self.atom_shuffle,
                debug=False
            )
        except (FunctionTimedOut, Exception):
            # Catch FunctionTimedOut and other rendering errors
            success = False
            graph = None

        if not success or graph is None:
            # Fallback: just return empty with a dummy image
            dummy_img = np.full((10, 10, 3), 255, dtype=np.uint8)
            dummy_img, _ = self.pad_to_square(dummy_img)
            img_tensor = self.post_transforms(image=dummy_img, keypoints=[])['image']
            return {'img': img_tensor, 
                    'atom_seq': [], 
                    'bond_mat': np.zeros((1, 1), dtype=int), 
                    'success': False}

        symbols = graph['symbols']
        coords = graph['coords']  # list of [x, y]
        bond_mat = graph['edges']
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Apply geo-transforms (rotation, etc.) with keypoints
        geo_result = self.geo_transforms(image=img, keypoints=coords)
        img = geo_result['image']
        coords = geo_result['keypoints']
        # Step 2: Pad to square (manual, handles keypoints)
        img, coords = self.pad_to_square(img, coords)
        # Step 3: Apply image transforms (blur, noise, etc.)
        img_result = self.img_transforms(image=img, keypoints=coords)
        img = img_result['image']
        coords = img_result['keypoints']
        # Step 4: Apply post-transforms (grayscale, normalize, to tensor)
        post_result = self.post_transforms(image=img, keypoints=coords)
        img_tensor = post_result['image']
        coords = post_result['keypoints']
        
        # Use target image size for binning
        H, W = self.image_size
        n_bins = self.vocab.n_bins
        # Binning - format: [symbol, x_bin, y_bin] per atom
        atom_seq = []
        for symbol, coord in zip(symbols, coords):
            x = min(max(0, coord[0]), W - 1)
            y = min(max(0, coord[1]), H - 1)

            x_bin = int((x / W) * n_bins) 
            y_bin = int((y / H) * n_bins)
            x_bin = max(0, min(n_bins - 1, x_bin))
            y_bin = max(0, min(n_bins - 1, y_bin))
            atom_seq.extend([symbol, x_bin, y_bin])

        return {
            'img': img_tensor,
            'atom_seq': atom_seq,
            'bond_mat': bond_mat,
            'success': True
        }

# ======================== Collate Function ========================

def collate_fn(batch_list: List[Dict], vocab: MolScannerVocab, max_atoms_limit: int = 100) -> Optional[Dict]:
    """
    Collate function with implicit atom extraction (finding X, Y pairs).
    Added max_atoms_limit to prevent OOM in BondPredictor.
    """
    # 1. Stack images
    batch_list = [item for item in batch_list if item is not None]
    if len(batch_list) == 0: return None
    
    # ===== 过滤掉失败样本和原子数过多的分子 =====
    filtered_batch = []
    for item in batch_list:
        # 1. 过滤失败样本 (success=False 或 atom_seq 为空)
        if not item.get('success', True) or len(item['atom_seq']) == 0:
            continue
        # 2. 过滤原子数过多的分子
        # atom_seq 结构: [x, y, symbol, x, y, symbol, ...]
        # 每3个元素代表一个原子
        n_atoms = len(item['atom_seq']) // 3
        if n_atoms <= max_atoms_limit:
            filtered_batch.append(item)
    
    if len(filtered_batch) == 0:
        return None
    
    batch_list = filtered_batch
    # ========================================
    
    batch_size = len(batch_list)
    images = torch.stack([item['img'] for item in batch_list])
    
    # 2. Tokenize
    tokenized_seqs = []
    for item in batch_list:
        tokens = vocab.tokenize_atom_seq(item['atom_seq'])
        tokenized_seqs.append(tokens)
    
    max_len = max(len(seq) for seq in tokenized_seqs)
    tgt_tokens = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long)
    tgt_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    
    for i, tokens in enumerate(tokenized_seqs):
        seq_len = len(tokens)
        tgt_tokens[i, :seq_len] = torch.tensor(tokens, dtype=torch.long)
        tgt_padding_mask[i, :seq_len] = False
    
    # 3. Pre-compute atom indices
    # Find the Y_BIN position for each atom (marks complete atom in [Symbol*, X, Y] format)
    # We look for positions where: current is Y_BIN
    atom_indices_list = []
    max_atoms = 1
    
    for tokens in tokenized_seqs:
        indices = []
        for i in range(len(tokens)):
            tok = tokens[i]
            # Current token is Y_BIN (end of atom)
            if vocab.is_y_coord_token(tok):
                indices.append(i)
        
        atom_indices_list.append(indices)
        max_atoms = max(max_atoms, len(indices))
    
    # Pad to global max_atoms
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
        ...
        """
        if self.backbone_name == 'resnet50':
            x = self.backbone(x)
        else:
            x = self.backbone(x)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        
        x = self.proj(x)
        return x

# ======================== Positional Encoding ========================

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for image features."""
    
    def __init__(self, d_model: int, max_h: int = 100, max_w: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # 确保 d_model 是 4 的倍数（分配给 y_sin, y_cos, x_sin, x_cos）
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4"
        
        pe = torch.zeros(max_h, max_w, d_model)
        
        y_pos = torch.arange(0, max_h).unsqueeze(1).float()  # [max_h, 1]
        x_pos = torch.arange(0, max_w).unsqueeze(1).float()  # [max_w, 1]
        
        # 每个维度分配 d_model/4 个频率
        dim_per_axis = d_model // 4
        div_term = torch.exp(torch.arange(0, dim_per_axis) * -(np.log(10000.0) / dim_per_axis))
        
        # 计算编码
        y_sin = torch.sin(y_pos * div_term)  # [max_h, dim_per_axis]
        y_cos = torch.cos(y_pos * div_term)  # [max_h, dim_per_axis]
        x_sin = torch.sin(x_pos * div_term)  # [max_w, dim_per_axis]
        x_cos = torch.cos(x_pos * div_term)  # [max_w, dim_per_axis]
        
        # 组合: [max_h, max_w, d_model]
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

# ======================== Atom Predictor (Transformer Decoder) ========================

class AtomPredictor(nn.Module):
    """
    Auto-regressive Transformer Decoder for atom sequence prediction.
    Optimized with causal mask caching and optional gradient checkpointing.
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


# ======================== Bond Predictor ========================

class BondPredictor(nn.Module):
    """
    Memory-efficient MLP-based bond predictor.
    Uses chunked processing to reduce peak VRAM usage for large molecules.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_bond_classes: int = 7,
        chunk_size: int = 32  # Process atoms in chunks to save VRAM
    ):
        """
        Args:
            d_model: Hidden dimension from atom predictor
            n_bond_classes: Number of bond types (0-6)
            chunk_size: Number of atom rows to process at once (lower = less VRAM)
        """
        super().__init__()
        self.d_model = d_model
        self.n_bond_classes = n_bond_classes
        self.chunk_size = chunk_size
        
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
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        atom_indices: torch.Tensor,
        atom_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Memory-efficient forward with optional chunked processing.
        
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
        
        # Extract atom representations efficiently using gather
        # Expand indices for gather: [B, N] -> [B, N, d_model]
        # Use expand().contiguous() to avoid memory issues with non-contiguous tensors
        expanded_indices = atom_indices.unsqueeze(-1).expand(B, N, dim).contiguous()
        atom_hidden = torch.gather(hidden_states, 1, expanded_indices)  # [B, N, d_model]
        
        # Decide whether to use chunked processing based on memory estimate
        # Peak memory ~ B * N * N * 2 * d_model * 4bytes (for float32)
        estimated_bytes = B * N * N * 2 * dim * 4
        use_chunked = N > self.chunk_size and estimated_bytes > 500_000_000  # >500MB
        
        if use_chunked:
            # Chunked processing for large molecules
            edge_logits = self._forward_chunked(atom_hidden, B, N, dim, device)
        else:
            # Standard processing for small molecules
            edge_logits = self._forward_standard(atom_hidden, B, N, dim)
        
        # Apply mask: create valid pair mask
        valid_pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)  # [B, N, N]
        diag_mask = self._get_diag_mask(N, device)
        valid_pair_mask = valid_pair_mask & ~diag_mask.unsqueeze(0)
        
        # Mask invalid positions (use -1e4 instead of -65000 to avoid fp16 overflow issues)
        edge_logits = edge_logits.masked_fill(~valid_pair_mask.unsqueeze(1), -1e4)
        
        return edge_logits
    
    def _forward_standard(self, atom_hidden: torch.Tensor, B: int, N: int, dim: int) -> torch.Tensor:
        """Standard pairwise computation - fast but uses more memory."""
        # Use broadcasting instead of expand to save memory
        atom_i = atom_hidden.unsqueeze(2)  # [B, N, 1, d]
        atom_j = atom_hidden.unsqueeze(1)  # [B, 1, N, d]
        
        # Compute features directly without expanding
        pair_sum = atom_i + atom_j  # broadcasts to [B, N, N, d]
        pair_diff = torch.abs(atom_i - atom_j)
        pair_features = torch.cat([pair_sum, pair_diff], dim=3)
        
        edge_logits = self.mlp(pair_features).permute(0, 3, 1, 2)
        return edge_logits
    
    def _forward_chunked(self, atom_hidden: torch.Tensor, B: int, N: int, dim: int, device: torch.device) -> torch.Tensor:
        """Chunked processing for large molecules - slower but uses less memory."""
        edge_logits = torch.empty(B, self.n_bond_classes, N, N, device=device, dtype=atom_hidden.dtype)
        
        for i in range(0, N, self.chunk_size):
            i_end = min(i + self.chunk_size, N)
            atom_i_chunk = atom_hidden[:, i:i_end].unsqueeze(2)  # [B, chunk, 1, d]
            atom_j = atom_hidden.unsqueeze(1)  # [B, 1, N, d]
            
            pair_sum = atom_i_chunk + atom_j
            pair_diff = torch.abs(atom_i_chunk - atom_j)
            pair_features = torch.cat([pair_sum, pair_diff], dim=3)
            
            chunk_logits = self.mlp(pair_features).permute(0, 3, 1, 2)
            edge_logits[:, :, i:i_end, :] = chunk_logits
        
        return edge_logits


# ======================== Helper: Extract Atom Indices ========================

def extract_atom_indices_from_tokens(
    tgt_tokens: torch.Tensor, 
    vocab: MolScannerVocab
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized extraction of atom indices from tokenized sequences.
    Strategy: In [Symbol*, X_BIN, Y_BIN] format, find the Y_BIN of each atom.
    
    Args:
        tgt_tokens: [B, T]
    Returns:
        atom_indices: [B, max_N]
        atom_counts: [B]
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
    # Get all (batch_idx, position) pairs where Y_BIN occurs
    y_bin_positions = torch.nonzero(is_y_bin, as_tuple=False)  # [num_y_bins, 2]
    
    if y_bin_positions.numel() > 0:
        # Group by batch and assign to output
        # Use scatter with cumulative counts
        batch_indices = y_bin_positions[:, 0]  # which batch
        positions = y_bin_positions[:, 1]      # position in sequence
        
        # Compute within-batch index for each Y_BIN token
        # Create cumsum per batch to get the index within each sequence
        ones = torch.ones(y_bin_positions.size(0), dtype=torch.long, device=device)
        batch_cumsum = torch.zeros(B + 1, dtype=torch.long, device=device)
        batch_cumsum.scatter_add_(0, batch_indices + 1, ones)
        batch_cumsum = batch_cumsum.cumsum(0)
        
        # Calculate in-batch indices
        global_idx = torch.arange(y_bin_positions.size(0), device=device)
        in_batch_idx = global_idx - batch_cumsum[batch_indices]
        
        # Assign positions to output tensor
        atom_indices[batch_indices, in_batch_idx] = positions
    
    return atom_indices, atom_counts

# ======================== MolScanModel ========================

class MolScannerModel(nn.Module):
    """Complete MolScanner model: Image → Token Sequence → Bond Matrix
    
    Optimized with:
    - Gradient checkpointing for reduced VRAM usage
    - Chunked bond prediction for large molecules
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
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        bond_chunk_size: int = 32
    ):
        """
        Args:
            vocab: MolScannerVocab instance
            image_size: Target image size (H, W)
            backbone: Encoder backbone name
            pretrained: Use pretrained backbone weights
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            use_gradient_checkpointing: Enable gradient checkpointing to save VRAM
            bond_chunk_size: Chunk size for bond predictor (lower = less VRAM)
        """
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
        
        # Atom Predictor with optional gradient checkpointing
        self.atom_predictor = AtomPredictor(
            vocab_size=len(vocab),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Bond Predictor with chunked processing support
        self.bond_predictor = BondPredictor(
            d_model=d_model,
            n_bond_classes=7,
            chunk_size=bond_chunk_size
        )
        
        # Inference transform: resize with padding
        self.inference_transform_list = [
        A.LongestMaxSize(max_size=self.image_size[0], interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(
            min_height=self.image_size[0], 
            min_width=self.image_size[1], 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=255
        ),
        A.ToGray(num_output_channels=3),  # Keep 3 channels for pretrained backbone
        A.Normalize(),  # ImageNet normalization
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
        """
        Forward pass with explicit atom mask.
        """
        # Defensive shape check for images
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(
                f"Expected images with shape [B, 3, H, W], got {images.shape}. "
                f"Ensure ToGray uses num_output_channels=3."
            )
        
        # Image encoding
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        # Flatten to sequence: [B, H*W, d_model]
        B, C, H, W = img_features.shape
        
        # Atom prediction
        T = tgt_tokens.size(1)

        # ===== 添加断言检查（而不是截断）=====
        max_pos = self.atom_predictor.pos_encoder.num_embeddings
        if T > max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {max_pos}. "
                f"This should be handled before calling forward()."
            )
        # ===================================

        hidden_states, token_logits = self.atom_predictor(
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
        
        # ===== multi-GPU atom sequence统一维度调整（合并填充和截断）=====
        current_size = atom_indices.size(1)
        
        if current_size != max_atoms:
            if current_size < max_atoms:
                # 填充
                padding_size = max_atoms - current_size
                atom_indices = F.pad(atom_indices, (0, padding_size), value=0)
                atom_mask = F.pad(atom_mask, (0, padding_size), value=False)
            else:
                # 截断（理论上不应该发生，但作为保护）
                atom_indices = atom_indices[:, :max_atoms]
                atom_mask = atom_mask[:, :max_atoms]
        
        atom_indices = atom_indices.to(images.device)
        atom_mask = atom_mask.to(images.device)
        assert atom_indices.size(1) == max_atoms, \
            f"atom_indices size mismatch: {atom_indices.size(1)} != {max_atoms}"
        assert atom_mask.size(1) == max_atoms, \
            f"atom_mask size mismatch: {atom_mask.size(1)} != {max_atoms}"
        # =======================
        
        # Predict bonds
        if max_atoms == 0:
            edge_logits_padded = torch.zeros((B, 7, 1, 1), device=images.device)
        else:
            # Defensive clamp: ensure all indices are within valid range for hidden_states
            T_hidden = hidden_states.size(1)
            atom_indices = atom_indices.clamp(0, T_hidden - 1)
            
            edge_logits_padded = self.bond_predictor(
                hidden_states, 
                atom_indices,
                atom_mask
            )
        
        return token_logits, edge_logits_padded, hidden_states

    def load_model(self, path: str, device: Optional[torch.device] = None):
        """
        Load trained model weights.
        
        Args:
            path: Path to the saved model checkpoint (.pth file)
            device: Device to load the model to (default: current device)
        
        Usage:
            model = MolScanModel(vocab=vocab, ...)
            model.load_model('path/to/best.pth')
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Load state dict
        state_dict = torch.load(path, map_location=device)
        
        # Load weights
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        
        print(f'Model loaded from: {path}')

# ======================== Inference Functions ========================

    def predict_step(self, img_features: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        """
        Predict next token logits given current sequence.
        
        Args:
            img_features: [B, d_model, H, W] (already pos-encoded)
            current_tokens: [B, T]
        Returns:
            next_token_logits: [B, vocab_size]
        """
        _, logits = self.atom_predictor(
            img_features=img_features,
            tgt_tokens=current_tokens,
            tgt_key_padding_mask=None
        )
        return logits[:, -1, :]

    def _apply_constraints(self, logits: torch.Tensor, last_token: int) -> torch.Tensor:
        """Apply structural constraints based on [Symbol*, X_BIN, Y_BIN] pattern."""
        logits = logits.clone()
        vocab = self.vocab
        
        # Check token types
        is_last_x_bin = vocab.is_x_coord_token(last_token)
        is_last_y_bin = vocab.is_y_coord_token(last_token)
        
        if last_token == vocab.sos_idx:
            # After SOS: expect Symbol char (start of first atom)
            # 禁止所有 coord bins 和特殊 tokens
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.eos_idx, vocab.unk_idx]] = float('-inf')
            
        elif is_last_x_bin:
            # After X_BIN: expect Y_BIN only
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = False
            logits.masked_fill_(mask, float('-inf'))
            
        elif is_last_y_bin:
            # After Y_BIN: expect Symbol char (new atom) or EOS
            # 禁止所有 coord bins 和其他特殊 tokens
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')
            
        else:
            # After Symbol char: expect more chars or X_BIN (end of symbol)
            # 禁止 Y_BIN (Y must follow X) 和特殊 tokens (EOS 也禁止，符号后必须有坐标)
            logits[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx, vocab.eos_idx]] = float('-inf')
            
        return logits

    def _greedy_decode(
        self, 
        feat: torch.Tensor, 
        max_len: int, 
        device: torch.device
    ) -> List[int]:
        """Greedy decoding for single image."""
        seq = [self.vocab.sos_idx]
        
        for _ in range(max_len):
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            logits = self.predict_step(feat, tgt)[0]
            
            last_token = seq[-1]
            logits = self._apply_constraints(logits, last_token)
            
            next_token = torch.argmax(logits).item()
            seq.append(next_token)  # Append first (including EOS)
            if next_token == self.vocab.eos_idx:
                break  # Then break
        
        return seq

    def _beam_search_decode(
        self, 
        feat: torch.Tensor, 
        beam_size: int, 
        max_len: int, 
        device: torch.device
    ) -> List[int]:
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
            if completed and beams and beams[0][1] < completed[-1][1]:
                break
        
        completed.extend(beams)
        return max(completed, key=lambda x: x[1])[0] if completed else [self.vocab.sos_idx]

    def _postprocess_sequence(
        self, 
        seq: List[int], 
        feat: torch.Tensor, 
        device: torch.device
    ) -> Dict:
        """Convert token sequence to atom_seq and predict bonds."""
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        
        # Extract atom indices
        atom_indices, atom_counts = extract_atom_indices_from_tokens(seq_tensor, self.vocab)
        atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
        
        # Get hidden states and predict bonds
        hidden_states, _ = self.atom_predictor(feat, seq_tensor)
        edge_logits = self.bond_predictor(hidden_states, atom_indices, atom_mask)
        edge_preds = torch.argmax(edge_logits, dim=1)[0].cpu().numpy()
        
        # Detokenize
        atom_seq_list, success = self.vocab.detokenize_atom_seq(seq)
        
        return {
            'token_ids': seq,
            'atom_seq': atom_seq_list,
            'bond_mat': edge_preds,
            'success': success
        }

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        beam_size: int = 1,
        max_len: int = 500,
        device: Optional[torch.device] = None
    ) -> List[Dict]:
        """Generate atom sequences and bond matrices for a batch of images."""
        self.eval()
        
        if device is None:
            device = images.device
        
        B = images.size(0)
        
        # Encode images once
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        results = []
        for b in range(B):
            feat = img_features[b:b+1]
            
            # Decode
            if beam_size == 1:
                seq = self._greedy_decode(feat, max_len, device)
            else:
                seq = self._beam_search_decode(feat, beam_size, max_len, device)
            
            # Postprocess
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
        if isinstance(image_source, str): # Path
            img_pil = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                img_pil = Image.fromarray(image_source).convert('RGB')
            elif image_source.shape[2] == 3:
                img_pil = Image.fromarray(image_source[:, :, ::-1])  # BGR -> RGB
            elif image_source.shape[2] == 4:
                img_pil = Image.fromarray(image_source[:, :, :3][:, :, ::-1])  # BGRA -> RGB
            else:
                raise ValueError(f"Unsupported numpy array shape: {image_source.shape}")
        elif isinstance(image_source, Image.Image):
            img_pil = image_source.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")
        
        return self.inference_transforms(image=np.array(img_pil))['image'].unsqueeze(0).to(device)

    def predict(
        self, 
        image_source,
        device: Optional[torch.device] = None,
        beam_size: int = 3,
        max_len: int = 500,
        return_preprocessed: bool = False
    ) -> Dict:
        """
        End-to-end prediction.
        
        Args:
            image_source: str (path), np.ndarray (BGR/Gray), PIL.Image, or torch.Tensor
            device: Inference device (default: model's device)
            beam_size: Beam search size (1 = greedy)
            max_len: Maximum sequence length
            return_preprocessed: Whether to include preprocessed image in result
        
        Returns:
            Dict with keys: token_ids, atom_seq, bond_mat, success, [preprocessed_img]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Preprocess
        if isinstance(image_source, torch.Tensor):
            img_tensor = self._preprocess_tensor(image_source, device)
        else:
            img_tensor = self._preprocess_image(image_source, device)
        
        # Generate
        result = self.generate(images=img_tensor, beam_size=beam_size, max_len=max_len, device=device)[0]
        
        if return_preprocessed:
            result['preprocessed_img'] = img_tensor.squeeze(0).cpu()
        
        return result

    def predict_batch(
        self,
        image_sources: List,
        device: Optional[torch.device] = None,
        beam_size: int = 3,
        max_len: int = 500
    ) -> List[Dict]:
        """Batch prediction on a single device."""
        
        # 1. Resolve Device
        if device is None:
            if list(self.parameters()):
                device = next(self.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        if isinstance(device, str):
            device = torch.device(device)

        # 2. Preprocess all images
        # We process to the target device directly, or CPU first if memory is tight
        tensors = []
        # Use CPU for preprocessing to avoid fragmentation, then move batch to GPU
        cpu_device = torch.device('cpu') 
        
        for src in image_sources:
            if isinstance(src, torch.Tensor):
                tensors.append(self._preprocess_tensor(src, cpu_device))
            else:
                tensors.append(self._preprocess_image(src, cpu_device))
        
        if not tensors:
            return []

        # 3. Create Batch and Move to Target Device
        img_batch = torch.cat(tensors, dim=0).to(device)

        # 4. Generate
        return self.generate(images=img_batch, beam_size=beam_size, max_len=max_len, device=device)


# ======================== Loss Computation ========================

def get_bond_class_weights():
    """
    Get predefined class weights for bond matrix.
    
    Bond types: {0: no bond, 1: single, 2: double, 3: triple, 4: aromatic, 5: solid wedge, 6: dash wedge}
    
    Args:
        n_classes: Number of bond classes (default 7)
    
    Returns:
        torch.Tensor of shape [n_classes]
    """
    # Predefined weights based on typical frequency (0=no bond is most common)
    # Give higher weight to actual bonds (1-6) compared to no bond (0)
    weights = torch.tensor([
        0.1,   # 0: no bond (very common, lower weight)
        1.0,   # 1: single bond
        2.0,   # 2: double bond (slightly more important)
        5.0,   # 3: triple bond (rare, higher weight)
        2.0,   # 4: aromatic bond
        5.0,   # 5: solid wedge (stereo, rare)
        5.0    # 6: dash wedge (stereo, rare)
    ], dtype=torch.float32)
    
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
    """
    Unified loss function with label smoothing for all token prediction.
    
    Key features:
    - Token prediction: cross entropy with label smoothing (unified for all tokens)
    - Bond prediction: weighted cross entropy with class weights
    """
    B, T, vocab_size = token_logits.shape
    device = token_logits.device
    
    # ===== 1. Token loss: unified cross entropy with label smoothing =====
    # Flatten for loss computation
    token_logits_flat = token_logits.reshape(-1, vocab_size)  # [B*T, vocab_size]
    tgt_tokens_flat = tgt_tokens.reshape(-1)  # [B*T]
    
    # Mask out padding tokens (use ignore_index)
    token_loss = F.cross_entropy(
        token_logits_flat, 
        tgt_tokens_flat, 
        ignore_index=vocab.pad_idx,
        label_smoothing=token_label_smoothing,
        reduction='mean'
    )
    
    # ===== 4. Bond loss =====
    # build a full batch target and let CrossEntropyLoss handle the global averaging.
    
    B, N_bond_type, N_atom_pred, _ = edge_logits.shape # [B, 7, N, N]
    
    # Initialize targets with -100 (standard ignore_index for PyTorch CrossEntropy)
    bond_targets = torch.full((B, N_atom_pred, N_atom_pred), -100, dtype=torch.long, device=device)
    
    # Fill in the valid bond matrices
    for b, bond_mat in enumerate(bond_matrices_list):
        N_atom_gt = bond_mat.shape[0]
        
        # Skip empty or invalid sized matrices
        if N_atom_gt == 0 or N_atom_gt > N_atom_pred:
            continue
            
        # Convert numpy bond_mat to tensor with proper dtype (long for cross entropy)
        bond_mat_tensor = torch.from_numpy(bond_mat.astype(np.int64)).to(device)
        
        # Assign to the target tensor
        bond_targets[b, :N_atom_gt, :N_atom_gt] = bond_mat_tensor
        
        # Mask the diagonal with -100 to strictly ignore self-loops
        bond_targets[b].fill_diagonal_(-100)

    # Check if we have any valid targets (not -100) to avoid NaN
    if (bond_targets != -100).any():
        # CrossEntropyLoss expects:
        # Input: [B, C, N, N]
        # Target: [B, N, N]
        # reduction='mean' averages over all valid (non-ignored) pixels in the batch
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

# ======================== Metrics Computation ========================

def compute_metrics(
    token_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    tgt_tokens: torch.Tensor,
    bond_matrices_list: List[np.ndarray],
    tgt_padding_mask: torch.Tensor,
    vocab: MolScannerVocab
) -> Dict[str, float]:
    """
    Compute evaluation metrics with separate X/Y coordinate analysis.
    """
    device = token_logits.device
    
    # ===== 1. Separate Symbol, X-coord, Y-coord tokens =====
    is_x_coord_mask = (tgt_tokens >= vocab.x_bin_start_idx) & (tgt_tokens <= vocab.x_bin_end_idx)
    is_y_coord_mask = (tgt_tokens >= vocab.y_bin_start_idx) & (tgt_tokens <= vocab.y_bin_end_idx)
    is_coord_mask = is_x_coord_mask | is_y_coord_mask
    is_symbol_mask = ~is_coord_mask & (tgt_tokens != vocab.pad_idx)
    
    # Valid mask (not padding)
    valid_mask = ~tgt_padding_mask 
    
    # ===== 2. Token Predictions =====
    token_preds = token_logits.argmax(dim=-1)  # [B, T]
    
    # ===== 3. Symbol Accuracy =====
    symbol_valid_mask = is_symbol_mask & valid_mask
    if symbol_valid_mask.any():
        correct_symbols = (token_preds == tgt_tokens) & symbol_valid_mask
        symbol_acc = correct_symbols.sum().item() / symbol_valid_mask.sum().item()
    else:
        symbol_acc = 0.0
    
    # ===== 4. Coord Accuracy (Combined and Separate X/Y) =====
    coord_valid_mask = is_coord_mask & valid_mask
    if coord_valid_mask.any():
        correct_coords = (token_preds == tgt_tokens) & coord_valid_mask
        coord_acc = correct_coords.sum().item() / coord_valid_mask.sum().item()
    else:
        coord_acc = 0.0
    
    # X coordinate accuracy and MAE
    x_valid_mask = is_x_coord_mask & valid_mask
    if x_valid_mask.any():
        correct_x = (token_preds == tgt_tokens) & x_valid_mask
        x_acc = correct_x.sum().item() / x_valid_mask.sum().item()
        
        # Mean Absolute Error in bins (only for predictions within X_BIN range)
        x_preds = token_preds[x_valid_mask]
        x_targets = tgt_tokens[x_valid_mask]
        
        # Only count predictions that are valid X_BIN tokens
        valid_x_preds_mask = (x_preds >= vocab.x_bin_start_idx) & (x_preds <= vocab.x_bin_end_idx)
        if valid_x_preds_mask.any():
            x_pred_bins = x_preds[valid_x_preds_mask] - vocab.x_bin_start_idx
            x_true_bins = x_targets[valid_x_preds_mask] - vocab.x_bin_start_idx
            x_mae = (x_pred_bins - x_true_bins).abs().float().mean().item()
        else:
            x_mae = float('nan')  # No valid predictions
    else:
        x_acc = 0.0
        x_mae = float('nan')
    
    # Y coordinate accuracy and MAE
    y_valid_mask = is_y_coord_mask & valid_mask
    if y_valid_mask.any():
        correct_y = (token_preds == tgt_tokens) & y_valid_mask
        y_acc = correct_y.sum().item() / y_valid_mask.sum().item()
        
        # Only count predictions that are valid Y_BIN tokens
        y_preds = token_preds[y_valid_mask]
        y_targets = tgt_tokens[y_valid_mask]
        
        valid_y_preds_mask = (y_preds >= vocab.y_bin_start_idx) & (y_preds <= vocab.y_bin_end_idx)
        if valid_y_preds_mask.any():
            y_pred_bins = y_preds[valid_y_preds_mask] - vocab.y_bin_start_idx
            y_true_bins = y_targets[valid_y_preds_mask] - vocab.y_bin_start_idx
            y_mae = (y_pred_bins - y_true_bins).abs().float().mean().item()
        else:
            y_mae = float('nan')
    else:
        y_acc = 0.0
        y_mae = float('nan')
    
    # ===== 5. Bond Accuracy (Full Matrix) =====
    all_bond_preds = []
    all_bond_targets = []
    
    for b, bond_mat in enumerate(bond_matrices_list):
        N_gt = bond_mat.shape[0]
        
        # Safety check: skip empty graphs or if prediction size is smaller than GT
        if N_gt == 0 or N_gt > edge_logits.shape[2]:
            continue
        
        bond_preds_sample = edge_logits[b, :, :N_gt, :N_gt].argmax(dim=0)
        bond_targets_sample = torch.from_numpy(bond_mat).to(device)
        
        off_diag_mask = ~torch.eye(N_gt, dtype=torch.bool, device=device)
        
        preds_flat = bond_preds_sample[off_diag_mask]
        targets_flat = bond_targets_sample[off_diag_mask]
        
        all_bond_preds.append(preds_flat.cpu())
        all_bond_targets.append(targets_flat.cpu())
    
    if len(all_bond_preds) > 0:
        all_bond_preds = torch.cat(all_bond_preds).numpy()
        all_bond_targets = torch.cat(all_bond_targets).numpy()
        bond_acc = float(np.mean(all_bond_preds == all_bond_targets))
    else:
        bond_acc = 0.0
    
    return {
        'symbol_acc': symbol_acc,
        'coord_acc': coord_acc,
        'x_coord_acc': x_acc,
        'y_coord_acc': y_acc,
        'x_coord_mae': x_mae,
        'y_coord_mae': y_mae,
        'bond_acc': bond_acc,
    }

# ======================== Sequence Truncation Helper ========================

def truncate_sequences(
    tgt_tokens: torch.Tensor,
    tgt_padding_mask: torch.Tensor,
    atom_indices: torch.Tensor,
    atom_mask: torch.Tensor,
    max_seq_len: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncate sequences to max_seq_len and update atom indices/mask accordingly.
    
    Args:
        tgt_tokens: [B, T]
        tgt_padding_mask: [B, T]
        atom_indices: [B, N]
        atom_mask: [B, N]
        max_seq_len: Maximum sequence length
    
    Returns:
        Truncated tensors
    """
    if tgt_tokens.size(1) <= max_seq_len:
        # Even without truncation, we need to account for teacher forcing shift (:-1)
        # which reduces effective sequence length by 1
        effective_max = tgt_tokens.size(1) - 1
        valid_atom_mask = atom_indices < effective_max
        atom_mask = atom_mask & valid_atom_mask
        # Clamp invalid indices to 0 to prevent out-of-bounds access during gather
        atom_indices = torch.where(valid_atom_mask, atom_indices, torch.zeros_like(atom_indices)).contiguous()
        return tgt_tokens, tgt_padding_mask, atom_indices, atom_mask.contiguous()
    
    # Truncate tokens and mask
    tgt_tokens = tgt_tokens[:, :max_seq_len].contiguous()
    tgt_padding_mask = tgt_padding_mask[:, :max_seq_len].contiguous()
    
    # Update atom_indices and atom_mask
    # Note: Account for teacher forcing shift (:-1) which reduces effective length by 1
    # After truncation and shift, valid indices are 0 to max_seq_len-2
    effective_max = max_seq_len - 1
    valid_atom_mask = atom_indices < effective_max
    atom_mask = atom_mask & valid_atom_mask
    # Clamp invalid indices to 0 to prevent out-of-bounds access during gather
    atom_indices = torch.where(valid_atom_mask, atom_indices, torch.zeros_like(atom_indices)).contiguous()
    
    return tgt_tokens, tgt_padding_mask, atom_indices, atom_mask.contiguous()

# ======================== Training Loop ========================

def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints.
    Returns 0.0 if either SMILES is invalid.
    """
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


def _evaluate_worker(
    rank: int, 
    world_size: int, 
    model_state_dict: dict,
    model_config: dict,
    data_list: List[dict],
    beam_size: int,
    return_dict: dict
):
    """
    Worker function for multi-GPU evaluation.
    Each worker processes its share of samples on a specific GPU.
    
    Args:
        rank: GPU index (0, 1, 2, ...)
        world_size: Total number of GPUs
        model_state_dict: Model weights (shared across processes)
        model_config: Model configuration dict
        data_list: List of dicts with 'img_path' and 'gt_smiles' keys
        beam_size: Beam search size
        return_dict: Multiprocessing Manager dict to store results
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    
    try:
        device = torch.device(f'cuda:{rank}')
        
        # Calculate data split
        chunk_size = math.ceil(len(data_list) / world_size)
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(data_list))
        my_data = data_list[start_idx:end_idx]
        
        if not my_data:
            return_dict[rank] = {'exact_matches': 0, 'tanimoto_scores': [], 
                                 'valid_count': 0, 'failed_count': 0}
            return
        
        # Recreate model on this GPU
        vocab = MolScannerVocab(n_bins=model_config['n_bins'])
        model = MolScannerModel(
            vocab=vocab,
            image_size=model_config['image_size'],
            backbone=model_config['backbone'],
            pretrained=False,  # We're loading weights, no need for pretrained
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_decoder_layers=model_config['num_decoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config.get('dropout', 0.1)
        )
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        
        exact_matches = 0
        tanimoto_scores = []
        valid_count = 0
        failed_count = 0
        
        # Process samples
        iterator = my_data
        if rank == 0:
            iterator = tqdm(my_data, desc=f'GPU {rank} Eval', leave=False)
        
        with torch.no_grad():
            for item in iterator:
                img_path = item['img_path']
                gt_smiles = item['gt_smiles']
                valid_count += 1
                
                try:
                    result = model.predict(img_path, device=device, beam_size=beam_size)
                    
                    if not result.get('success', False) or not result.get('atom_seq'):
                        failed_count += 1
                        tanimoto_scores.append(0.0)
                        continue
                    
                    # Convert prediction to SMILES
                    atom_seq = result['atom_seq']
                    n = len(atom_seq) // 3
                    symbols = [atom_seq[k * 3] for k in range(n)]
                    coords = [(atom_seq[k * 3 + 1], atom_seq[k * 3 + 2]) for k in range(n)]
                    
                    pred_smiles, _, _, conv_success = _convert_graph_to_smiles(
                        coords=coords, symbols=symbols, edges=result['bond_mat']
                    )
                    
                    if not conv_success or pred_smiles is None:
                        failed_count += 1
                        tanimoto_scores.append(0.0)
                        continue
                    
                    pred_smiles, pred_success = canonicalize_smiles(pred_smiles, ignore_cistrans=True)
                    if not pred_success:
                        failed_count += 1
                        tanimoto_scores.append(0.0)
                        continue
                    
                    # Check exact match
                    if pred_smiles == gt_smiles:
                        exact_matches += 1
                    
                    # Compute Tanimoto similarity
                    tanimoto = compute_tanimoto_similarity(gt_smiles, pred_smiles)
                    tanimoto_scores.append(tanimoto)
                    
                except Exception:
                    failed_count += 1
                    tanimoto_scores.append(0.0)
        
        return_dict[rank] = {
            'exact_matches': exact_matches,
            'tanimoto_scores': tanimoto_scores,
            'valid_count': valid_count,
            'failed_count': failed_count
        }
        
        if rank == 0:
            print(f'GPU {rank}: processed {valid_count} samples')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"GPU {rank} Failed: {e}")
        return_dict[rank] = {'exact_matches': 0, 'tanimoto_scores': [], 
                             'valid_count': 0, 'failed_count': 0}


def evaluate_on_benchmark(
    model: nn.Module,
    benchmark_dir: str,
    benchmark_csv_path: str,
    vocab: 'MolScannerVocab',
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    beam_size: int = 1,
    max_samples: Optional[int] = None,
    use_multi_gpu: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on a real-world benchmark dataset with optional multi-GPU acceleration.
    
    Args:
        model: The MolScannerModel to evaluate
        benchmark_dir: Directory containing benchmark images (e.g., '.../benchmark/real/USPTO')
        benchmark_csv_path: Path to CSV file with 'image_id' and 'SMILES' columns
        vocab: MolScannerVocab instance
        device: Device to run inference on (used for single-GPU mode)
        epoch: Current epoch number for logging
        writer: TensorBoard SummaryWriter
        global_step: Global training step for logging
        beam_size: Beam size for decoding (default: 1 for greedy)
        max_samples: If set, limit evaluation to this many samples (for faster validation)
        use_multi_gpu: Whether to use multi-GPU acceleration when available (default: True)
    
    Returns:
        Dict with 'exact_match_acc' and 'avg_tanimoto' metrics
    """
    model.eval()
    
    # Load benchmark data
    label_df = pd.read_csv(benchmark_csv_path)
    if max_samples is not None and max_samples < len(label_df):
        label_df = label_df.sample(n=max_samples, random_state=42)
    
    print(f'\nEvaluating on benchmark: {benchmark_csv_path}')
    print(f'Total samples: {len(label_df)}')
    
    # Get the actual model (unwrap DataParallel if needed)
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Check multi-GPU availability
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = use_multi_gpu and n_gpus > 1
    
    if use_multi_gpu:
        print(f'Using {n_gpus} GPUs for evaluation')
    else:
        print(f'Using single device: {device}')
    
    # Preprocess data: filter valid samples and prepare data list
    data_list = []
    for _, row in label_df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(benchmark_dir, f"{img_id}.png")
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue
        
        # Process ground truth SMILES
        try:
            raw_gt_smiles = row['SMILES']
            gt_smiles = remove_atom_mapping(raw_gt_smiles)
            gt_smiles, gt_success = canonicalize_smiles(gt_smiles, ignore_cistrans=True)
            if not gt_success:
                continue
        except Exception:
            continue
        
        data_list.append({'img_path': img_path, 'gt_smiles': gt_smiles})
    
    print(f'Valid samples after filtering: {len(data_list)}')
    
    if len(data_list) == 0:
        print('No valid samples found!')
        return {
            'exact_match_acc': 0.0,
            'avg_tanimoto': 0.0,
            'valid_samples': 0,
            'failed_predictions': 0
        }
    
    # ===== Multi-GPU Evaluation =====
    if use_multi_gpu:
        # Prepare model config for workers
        model_config = {
            'n_bins': vocab.n_bins,
            'image_size': actual_model.image_size,
            'backbone': actual_model.image_encoder.backbone_name,
            'd_model': actual_model.d_model,
            'nhead': actual_model.atom_predictor.transformer_decoder.layers[0].self_attn.num_heads,
            'num_decoder_layers': len(actual_model.atom_predictor.transformer_decoder.layers),
            'dim_feedforward': actual_model.atom_predictor.transformer_decoder.layers[0].linear1.out_features,
        }
        
        # Get model state dict (move to CPU to share across processes)
        model_state_dict = {k: v.cpu() for k, v in actual_model.state_dict().items()}
        
        # ===== CRITICAL: Free GPU memory before spawning evaluation workers =====
        # Move the training model to CPU temporarily to free GPU memory
        original_device = device
        model.cpu()
        
        # Clear CUDA cache on all GPUs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for gpu_id in range(n_gpus):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        # ========================================================================
        
        # Use spawn context for CUDA compatibility
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(
                target=_evaluate_worker,
                args=(rank, n_gpus, model_state_dict, model_config, 
                      data_list, beam_size, return_dict)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes
        for p in processes:
            p.join()
        
        # Aggregate results
        exact_matches = 0
        tanimoto_scores = []
        valid_gt_count = 0
        failed_predictions = 0
        
        for rank in range(n_gpus):
            result = return_dict.get(rank, {})
            exact_matches += result.get('exact_matches', 0)
            tanimoto_scores.extend(result.get('tanimoto_scores', []))
            valid_gt_count += result.get('valid_count', 0)
            failed_predictions += result.get('failed_count', 0)
        
        # ===== Restore model to original device after multi-GPU evaluation =====
        model.to(original_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ========================================================================
    
    # ===== Single-GPU Evaluation =====
    else:
        exact_matches = 0
        tanimoto_scores = []
        valid_gt_count = 0
        failed_predictions = 0
        
        with torch.no_grad():
            for item in tqdm(data_list, desc=f'Epoch {epoch} [Benchmark Val]'):
                img_path = item['img_path']
                gt_smiles = item['gt_smiles']
                valid_gt_count += 1
                
                try:
                    result = actual_model.predict(img_path, device=device, beam_size=beam_size)
                    
                    if not result.get('success', False) or not result.get('atom_seq'):
                        failed_predictions += 1
                        tanimoto_scores.append(0.0)
                        continue
                    
                    # Convert prediction to SMILES
                    atom_seq = result['atom_seq']
                    n = len(atom_seq) // 3
                    symbols = [atom_seq[k * 3] for k in range(n)]
                    coords = [(atom_seq[k * 3 + 1], atom_seq[k * 3 + 2]) for k in range(n)]
                    
                    pred_smiles, _, _, conv_success = _convert_graph_to_smiles(
                        coords=coords, symbols=symbols, edges=result['bond_mat']
                    )
                    
                    if not conv_success or pred_smiles is None:
                        failed_predictions += 1
                        tanimoto_scores.append(0.0)
                        continue
                    
                    pred_smiles, pred_success = canonicalize_smiles(pred_smiles, ignore_cistrans=True)
                    if not pred_success:
                        failed_predictions += 1
                        tanimoto_scores.append(0.0)
                        continue
                    
                    # Check exact match
                    if pred_smiles == gt_smiles:
                        exact_matches += 1
                    
                    # Compute Tanimoto similarity
                    tanimoto = compute_tanimoto_similarity(gt_smiles, pred_smiles)
                    tanimoto_scores.append(tanimoto)
                    
                except Exception:
                    failed_predictions += 1
                    tanimoto_scores.append(0.0)
    
    # Compute metrics
    exact_match_acc = exact_matches / valid_gt_count if valid_gt_count > 0 else 0.0
    avg_tanimoto = np.mean(tanimoto_scores) if tanimoto_scores else 0.0
    
    # Print results
    print(f'Epoch {epoch} [Benchmark Val]:')
    print(f'  Valid GT samples: {valid_gt_count}')
    print(f'  Failed predictions: {failed_predictions}')
    print(f'  Exact Match Accuracy: {exact_match_acc:.4f} ({exact_matches}/{valid_gt_count})')
    print(f'  Average Tanimoto Similarity: {avg_tanimoto:.4f}')
    
    # Log to TensorBoard
    writer.add_scalar('Val/exact_match_acc', exact_match_acc, global_step)
    writer.add_scalar('Val/avg_tanimoto', avg_tanimoto, global_step)
    writer.add_scalar('Val/valid_samples', valid_gt_count, global_step)
    writer.add_scalar('Val/failed_predictions', failed_predictions, global_step)
    
    return {
        'exact_match_acc': exact_match_acc,
        'avg_tanimoto': avg_tanimoto,
        'valid_samples': valid_gt_count,
        'failed_predictions': failed_predictions
    }


def train(
    smiles_list: List[str],
    save_path: str,
    benchmark_dir: str,
    benchmark_csv_path: str,
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 4e-4,
    warmup_ratio: float = 0.05,  # warmup ratio
    smiles_num: int = int(1e6),
    max_atoms: int = 100,
    mol_augment: bool = True,
    train_atom_shuffle: bool = True,
    image_size: Tuple[int, int] = (384, 384),
    n_bins: int = 64,
    backbone: str = 'swin_b',
    pretrained: bool = True,
    d_model: int = 512,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    seed: int = 2025,
    early_stopping_patience: int = 5,
    val_max_samples: Optional[int] = None,
    # ===== optimization parameters =====
    num_workers: int = 4,
    use_amp: bool = True,
    use_gradient_checkpointing: bool = False,
    bond_chunk_size: int = 32,
    force_cpu: bool = False,
) -> List[float]:
    """
    Main training function with end-of-epoch benchmark validation.
    
    Optimizations available:
    - AMP (automatic mixed precision) for faster training
    - Gradient checkpointing for reduced VRAM usage
    - Chunked bond prediction for large molecules
    
    Args:
        smiles_list: List of SMILES strings for training
        save_path: Directory to save checkpoints and logs
        benchmark_dir: Directory containing benchmark images (e.g., 'data/benchmark/real/USPTO')
        benchmark_csv_path: Path to benchmark CSV with 'image_id' and 'SMILES' columns
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        warmup_ratio: Ratio of warmup steps to total steps
        smiles_num: Maximum number of SMILES to use for training
        max_atoms: Maximum number of atoms per molecule
        mol_augment: Whether to apply molecule augmentation
        train_atom_shuffle: Whether to shuffle atom order during training
        image_size: Target image size (H, W)
        n_bins: Number of coordinate bins
        backbone: Encoder backbone ('resnet50', 'swin_t', 'swin_b', etc.)
        pretrained: Whether to use pretrained backbone weights
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension in transformer
        dropout: Dropout rate
        num_workers: Number of data loading workers
        seed: Random seed
        use_amp: Whether to use automatic mixed precision
        early_stopping_patience: Number of epochs without improvement before stopping
        val_max_samples: Maximum number of validation samples (None = use all)
        force_cpu: Force CPU mode
        use_gradient_checkpointing: Enable gradient checkpointing to reduce VRAM (~30% savings)
        bond_chunk_size: Chunk size for bond predictor (lower = less VRAM, default 32)
    """

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Device setup
    if force_cpu:
        device = torch.device('cpu')
        use_multi_gpu = False
        use_amp = False  # CPU does not support AMP
        print('Forcing CPU mode')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    
    if use_multi_gpu:
        print(f'Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')
    else:
        print(f'Using device: {device}')
    
    # Print optimization settings
    print(f'Optimizations: AMP={use_amp}, GradCheckpoint={use_gradient_checkpointing}, '
          f'BondChunkSize={bond_chunk_size}')
    
    # Create vocab
    vocab = MolScannerVocab(n_bins=n_bins)
    print(f'Vocab size: {len(vocab)}')
    
    # Prepare training data
    n = len(smiles_list)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    indices = indices[:smiles_num]  # limit to smiles_num for training
    train_smiles = [smiles_list[i] for i in indices]
    
    print(f'Train size: {len(train_smiles)}')
    print(f'Benchmark validation: {benchmark_csv_path}')
    
    # Training dataset
    train_dataset = MoleculeDataset(
        smiles_list=train_smiles,
        shuffle_smiles=True,
        vocab=vocab,
        image_size=image_size,
        mol_augment=mol_augment,
        geo_augment=True,
        img_augment=True,
        atom_shuffle=train_atom_shuffle,
    )
    
    # Get predefined bond class weights
    bond_class_weights = get_bond_class_weights()
    
    # DataLoader
    dl_common = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'collate_fn': lambda x: collate_fn(x, vocab, max_atoms_limit=max_atoms)
    }
    if num_workers > 0:
        dl_common.update({'prefetch_factor': 4, 'persistent_workers': True})
    
    train_loader = DataLoader(train_dataset, shuffle=False, **dl_common)
    
    # Model with optimization flags
    model = MolScannerModel(
        vocab=vocab,
        backbone=backbone,
        pretrained=pretrained,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_gradient_checkpointing=use_gradient_checkpointing,
        bond_chunk_size=bond_chunk_size,
    ).to(device)
    
    # only enable DataParallel when using multiple GPUs
    if use_multi_gpu and device.type == 'cuda':
        model = nn.DataParallel(model)
    
    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # ===== 计算总步数和 warmup 步数 =====
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}%)')
    
    # ===== 创建 Warmup + Cosine Decay 学习率调度器 =====
    def lr_lambda(current_step: int) -> float:
        """
        Learning rate schedule:
        - Linear warmup for warmup_steps
        - Cosine decay after warmup
        """
        if current_step < warmup_steps:
            # Linear warmup: from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: from 1 to near 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    
    # TensorBoard
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(save_path, 'logs', timestamp)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Training loop with end-of-epoch validation
    best_val_acc = 0.0
    epochs_no_improve = 0
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in train_bar:
            if batch is None:
                continue
            
            # Ensure all tensors are contiguous for DataParallel
            images = batch['images'].to(device).contiguous()
            tgt_tokens = batch['tgt_tokens'].to(device).contiguous()
            tgt_padding_mask = batch['tgt_padding_mask'].to(device).contiguous()
            atom_indices = batch['atom_indices'].to(device).contiguous()
            atom_mask = batch['atom_mask'].to(device).contiguous()
            max_atoms = batch['max_atoms']
            bond_matrices_list = batch['bond_matrices_list']

            # ===== use helper function to truncate =====
            tgt_tokens, tgt_padding_mask, atom_indices, atom_mask = truncate_sequences(
                tgt_tokens, tgt_padding_mask, atom_indices, atom_mask, max_seq_len=1000
            )
            # ==========================
            
            optimizer.zero_grad()
            
            # Teacher forcing - ensure contiguous tensors
            tgt_input = tgt_tokens[:, :-1].contiguous()
            tgt_target = tgt_tokens[:, 1:].contiguous()
            tgt_input_mask = tgt_padding_mask[:, :-1].contiguous()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    token_logits, edge_logits, _ = model(
                        images=images,
                        tgt_tokens=tgt_input,
                        tgt_key_padding_mask=tgt_input_mask,
                        atom_indices=atom_indices,
                        atom_mask=atom_mask,
                        max_atoms=max_atoms
                    )
                    
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
                token_logits, edge_logits, _ = model(
                    images=images,
                    tgt_tokens=tgt_input,
                    tgt_key_padding_mask=tgt_input_mask,
                    atom_indices=atom_indices,
                    atom_mask=atom_mask,
                    max_atoms=max_atoms
                )
                
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
            
            # ===== update learning rate =====
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # ==========================
            
            running_loss += losses['total_loss'].item()
            num_batches += 1
            
            train_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.2f}",
                't': f"{losses['token_loss'].item():.2f}",
                'b': f"{losses['bond_loss'].item():.2f}",
            })
            
            writer.add_scalar('Loss/train', losses['total_loss'].item(), global_step)
            writer.add_scalar('Loss/train_token', losses['token_loss'].item(), global_step)
            writer.add_scalar('Loss/train_bond', losses['bond_loss'].item(), global_step)
            writer.add_scalar('LR/learning_rate', current_lr, global_step)
            global_step += 1
        
        # End of epoch summary
        avg_loss = running_loss / max(num_batches, 1)
        print(f'Epoch {epoch} [Train] - avg_loss: {avg_loss:.4f}')
        
        # ======================== End-of-Epoch Benchmark Validation ========================
        val_metrics = evaluate_on_benchmark(
            model=model,
            benchmark_dir=benchmark_dir,
            benchmark_csv_path=benchmark_csv_path,
            vocab=vocab,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            beam_size=1,
            max_samples=val_max_samples
        )
        
        val_acc = val_metrics['exact_match_acc']
        
        # Save best model based on exact match accuracy
        if val_acc > best_val_acc:
            epochs_no_improve = 0
            best_val_acc = val_acc
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), os.path.join(save_path, 'best.pth'))
            print(f'New best model saved (exact_match_acc={best_val_acc:.4f}).')
        else:
            epochs_no_improve += 1
            print(f'No improvement for {epochs_no_improve} epoch(s).')
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            # Save final before stopping
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), os.path.join(save_path, 'final.pth'))
            writer.close()
            return []
        
        # Save epoch checkpoint
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
        
        model.train()  # Switch back to train mode
    
    # Final save
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), os.path.join(save_path, 'final.pth'))
    writer.close()
    
    print('Training complete!')
    return []

def run_inference_worker(rank, world_size, model: MolScannerModel, data_paths, return_dict):
    """Worker function that runs in a separate process on a specific GPU."""
    
    # 1. Avoid CPU overload from OpenCV spawning too many threads per process
    cv2.setNumThreads(0)
    
    try:
        device = torch.device(f'cuda:{rank}')
        
        # 2. Calculate data split
        chunk_size = math.ceil(len(data_paths) / world_size)
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(data_paths))
        my_paths = data_paths[start_idx:end_idx]
        
        if not my_paths:
            return

        print(f"Rank {rank}: Processing {len(my_paths)} images on {device}...")
        
        # 3. Setup Model
        model.to(device)
        model.eval()

        # 4. Process in Mini-Batches
        # Process 128 images at a time to keep memory low and speed high
        mini_batch_size = 128 
        results = []
        
        # Only show progress bar on Rank 0 to avoid messy logs
        iterator = range(0, len(my_paths), mini_batch_size)
        if rank == 0:
            iterator = tqdm(iterator, desc=f"Rank {rank} Inference", total=len(iterator))
            
        for i in iterator:
            batch_paths = my_paths[i : i + mini_batch_size]
            
            # predict_batch will now handle just this small chunk
            batch_res = model.predict_batch(batch_paths, device=device, beam_size=1)
            results.extend(batch_res)
        
        return_dict[rank] = results
        print(f"Rank {rank}: Done.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Rank {rank} Failed: {e}")