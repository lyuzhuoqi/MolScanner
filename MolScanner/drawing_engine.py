from indigo import Indigo
from indigo.renderer import IndigoRenderer

import random
from typing import Dict, Any
import numpy as np
import string
import re
import cv2

cv2.setNumThreads(1)

from func_timeout import func_set_timeout

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from constants import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, ABBREVIATIONS, COLORS
from chemistry import _verify_chirality

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_WAVY_BOND_PROB = 0.2
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


# ----------------------Functions for molecule augmentation--------------------------
def add_explicit_hydrogen(mol):
    """
    Randomly make implicit hydrogens explicit in the molecule
    """

    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol


def add_rgroup(mol, smiles):
    """
    Randomly replace an implicit H atom with a R-group as a pseudo atom
    """

    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles:
        if random.random() < INDIGO_RGROUP_PROB:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol


def get_rand_symb():
    symb = random.choice(ELEMENTS)
    # Optional: Add lowercase to simulate two-letter elements or organic abbreviations
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    # Optional: Add uppercase (less common in standard formulas but adds noise)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    # Recursive group generation (e.g., (COOH)2)
    if random.random() < 0.1:
        # Recursion is fine, but we trust gen_rand_condensed to handle the content
        symb = f'({gen_rand_condensed()})'
    return symb


def get_rand_num():
    """
    Returns a number string or empty string.
    """
    if random.random() < 0.9:
        # 80% chance of empty string (implicit 1) inside this block
        if random.random() < 0.8:
            return ''
        else:
            return str(random.randint(2, 9))
    else:
        # 10% chance of double digits
        return '1' + str(random.randint(0, 9))


def gen_rand_condensed():
    tokens = []
    for i in range(5):
        symb = get_rand_symb()
        num = get_rand_num()
        
        tokens.append(symb)
        tokens.append(num)
        
        # Check if what we have so far is just a "single atom"
        # It is a single atom if:
        # 1. We are on the first iteration (i==0)
        # 2. No number was generated (num == '')
        # 3. It's not a complex group (no parenthesis in symb)
        is_single_atom = (i == 0) and (num == '') and ('(' not in symb)
        
        if is_single_atom:
            # If it's just a single atom (e.g., "C"), we FORCE the loop 
            # to continue to the next iteration to add another part.
            # This ensures we get at least "CH" or "CBr" instead of just "C".
            continue

        # Standard stop condition:
        # If we are past the first valid chunk, we have an 80% chance to stop.
        if i >= 1 and random.random() < 0.8:
            break
        # If we are at i==0 but we had a number (e.g. "C2"), we can also stop.
        elif i == 0 and not is_single_atom and random.random() < 0.8:
            break

    return ''.join(tokens)


def add_rand_condensed(mol):
    """
    Randomly replace a implicit H atom with random condensed formula as pseudo atoms
    """

    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
            
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol

def add_functional_group(indigo, mol, debug=False):
    ''' Randomly replace functional groups as pseudo atoms with their abbreviations '''

    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol

def add_color(indigo, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption('render-highlight-color-enabled', True)
            indigo.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol

def add_wavy_bond(mol, debug=False):
    ''' Randomly replace a wedge bond with wavy bond '''
    atoms = []
    for atom in mol.iterateStereocenters():
        atoms.append(atom)
    if len(atoms) > 0 and (random.random() < INDIGO_WAVY_BOND_PROB or debug):
        atom = random.choice(atoms)
        atom.changeStereocenterType(4)
    return mol

# --------------------------------------------------------------------------------------


# ----------------------Functions for graph and SMILES generation-----------------------
def get_atom_smiles(atom) -> str:
    """
    Generates a descriptive SMILES-like string for an Indigo Atom.
    """
    # To get ALL hydrogens (explicit and implicit)
    try:
        h_count = atom.countImplicitHydrogens()
    except:
        h_count = 0

    symbol = atom.symbol()

    # Check if we can use a simple representation (e.g., "C" instead of "[CH4]")
    is_in_organic_subset = symbol in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', '*']
    
    # An atom needs brackets if it's not in the organic subset, or if it has special properties.
    if not is_in_organic_subset or atom.isotope() or atom.charge() != 0 or h_count > 0:
        # Build the full descriptive string
        isotope_str = str(atom.isotope()) if atom.isotope() else ""
        h_str = f"H{h_count}" if h_count > 1 else ("H" if h_count == 1 else "")
        charge = atom.charge()
        charge_str = f"+{charge}" if charge > 0 else (str(charge) if charge < 0 else "")
        if charge == 1: charge_str = "+"
        if charge == -1: charge_str = "-"
        
        return f"[{isotope_str}{symbol}{h_str}{charge_str}]"
    
    return symbol


def get_graph(mol, image, n_bins=None):
    '''
    Get the graph representation of the molecule.
    '''

    coords, symbols = [], []
    index_map = {}

    # Atom
    atoms = [atom for atom in mol.iterateAtoms()]
    h, w, _ = image.shape
    for i, atom in enumerate(atoms):
        # Coordinates
        x, y = atom.coords()

        if n_bins is not None: # Binning
            x_bin = int((x / w) * n_bins)
            y_bin = int((y / h) * n_bins)
            x_bin = max(0, min(n_bins - 1, x_bin))
            y_bin = max(0, min(n_bins - 1, y_bin))
            coords.append([x_bin, y_bin])
        else: # No binning
            coords.append([x, y])
        # Symbols
        symbols.append(get_atom_smiles(atom))
        index_map[atom.index()] = i

    # Edges
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {
        'coords': coords,
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph


def _graph_to_molblock(graph: Dict[str, Any]) -> str:
    """Convert a molecule graph dict to an RDKit MolBlock string.

    Rebuilds the molecule from symbols / edges / coords via RDKit, verifies
    chirality, and returns the V2000 MolBlock.  Shared by
    ``_augment_graph_and_render`` and reusable by other graph pipelines.
    """
    from rdkit import Chem
    from chemistry import _verify_chirality
    from constants import RGROUP_SYMBOLS, ABBREVIATIONS

    mol = Chem.RWMol()
    symbols = graph['symbols']
    edges = graph['edges']
    coords = graph['coords']
    n = len(symbols)
    ids = []

    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom('*')
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom('*')
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except Exception:
                atom = Chem.Atom('*')
                Chem.SetAtomAlias(atom, symbol)
        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)
        ids.append(mol.AddAtom(atom))

    for i in range(n):
        for j in range(i + 1, n):
            order = int(edges[i][j])
            if order == 0:
                continue
            if order in (1, 5, 6):
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                if order == 5:
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(
                        Chem.BondDir.BEGINWEDGE)
                elif order == 6:
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(
                        Chem.BondDir.BEGINDASH)
            elif order == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif order == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif order == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)

    mol = _verify_chirality(mol, coords, edges, False)
    return Chem.MolToMolBlock(mol)


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if '*' in smiles:
        part_a, part_b = smiles.split(' ', maxsplit=1)
        m = re.search(r'\$(.*)\$', part_b)
        part_b = m.group(1) if m else ''
        symbols = [t for t in part_b.split(';') if len(t) > 0]
        output = ''
        cnt = 0
        for i, c in enumerate(part_a):
            if c != '*':
                output += c
            else:
                output += f'[{symbols[cnt]}]'
                cnt += 1
        return mol, output
    else:
        if ' ' in smiles:
            # special cases with extension
            smiles = smiles.split(' ')[0]
        return mol, smiles
# --------------------------------------------------------------------------------------


# ---------------------- Functions for Augmentation ------------------------------------
def _generate_style_config(mol, default_drawing_style=False):
    """
    Generate drawing style configuration dict for Indigo rendering.
    Returns a dict with all render options that can be applied to any Indigo session.
    """
    indigo_option_config: dict = {
        'render-output-format': 'png',
        'render-background-color': '1,1,1',
        'render-stereo-style': 'none',
        'render-label-mode': 'hetero',
        'render-font-family': 'Arial',
        'render-valences-visible': False,
    }

    mol_config: dict = {
        'mol.dearomatize': False,
    }

    if not default_drawing_style:
        thickness = random.uniform(0.5, 2)
        indigo_option_config['render-relative-thickness'] = thickness
        indigo_option_config['render-bond-line-width'] = random.uniform(1, 4 - thickness)
        
        if random.random() < 0.5:
            indigo_option_config['render-font-family'] = random.choice(['Arial', 'Times', 'Courier', 'Helvetica'])
        
        indigo_option_config['render-label-mode'] = random.choice(['hetero', 'terminal-hetero'])
        indigo_option_config['render-implicit-hydrogens-visible'] = random.choice([True, False])
        
        if random.random() < 0.1:
            indigo_option_config['render-stereo-style'] = 'old'
        if random.random() < 0.2:
            indigo_option_config['render-atom-ids-visible'] = True
        
        if random.random() < INDIGO_COMMENT_PROB:
            indigo_option_config['render-comment'] = str(random.randint(1, 20)) + random.choice(string.ascii_letters)
            indigo_option_config['render-comment-font-size'] = random.randint(40, 60)
            indigo_option_config['render-comment-alignment'] = random.choice([0, 0.5, 1])
            indigo_option_config['render-comment-position'] = random.choice(['top', 'bottom'])
            indigo_option_config['render-comment-offset'] = random.randint(2, 30)

        if random.random() < INDIGO_DEARMOTIZE_PROB: # The way aromatic rings are represented
            mol_config['mol.dearomatize'] = True
        else:
            mol_config['mol.dearomatize'] = False

    return mol, indigo_option_config, mol_config


def _apply_style_config(indigo, indigo_option_config, mol, mol_config):
    """Apply style configuration dict to an Indigo session."""
    for key, value in indigo_option_config.items():
        indigo.setOption(key, value)
    if mol_config['mol.dearomatize']:
        mol.dearomatize()
    else:
        mol.aromatize()
# --------------------------------------------------------------------------------------


# -------------------------------- Drawing Functions -----------------------------------
def _blank_image() -> np.ndarray:
    """Return a small blank white image as fallback."""
    return np.full((10, 10, 3), 255, dtype=np.uint8)


@func_set_timeout(5)  # Set timeout of 5 seconds
def generate_image_from_smiles(smiles, n_bins=None,
                               mol_augment=True, include_condensed=True, 
                               default_drawing_style=False, 
                               debug=False):
    '''
    Generate molecule image and graph from SMILES string.
    Args:
        smiles: SMILES string
        n_bins: int or None, number of bins for coordinate binning
        mol_augment: bool, whether to perform molecule augmentation
        include_condensed: bool, whether to include condensed formula in augmentation. Only used if mol_augment is True.
        default_drawing_style: bool, whether to use default drawing style
        debug: If True, raise exceptions instead of returning fallback
    Returns:
        img: np.ndarray (H, W, 3) Color image
        smiles: SMILES string (possibly modified by augmentation)
        graph: dict with 'symbols', 'edges', 'coords'
        success: bool, whether the generation was successful
        style_config: dict, drawing style configuration used
        mol_config: dict, molecule configuration used
    '''
    indigo = None
    renderer = None
    style_config, mol_config = None, None

    try:
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)

        mol = indigo.loadMolecule(smiles)
        # if mol.countAtoms() > 100:  # Skip huge molecules
        #     raise ValueError("Molecule too large")

        # Drawing Style Augmentation
        mol, style_config, mol_config = _generate_style_config(mol, default_drawing_style)
        _apply_style_config(indigo, style_config, mol, mol_config)

        # Molecule Augmentation
        if mol_augment:
            smiles = mol.canonicalSmiles()
            mol = add_explicit_hydrogen(mol)
            mol = add_rgroup(mol, smiles)
            if include_condensed:
                mol = add_rand_condensed(mol)
            mol = add_functional_group(indigo, mol, debug)
            mol = add_wavy_bond(mol, debug=debug)
            mol = add_color(indigo, mol)
            mol, smiles = generate_output_smiles(indigo, mol)
        
        mol.layout()
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1) # BGR format
        graph = get_graph(mol, img, n_bins=n_bins)
        success = True
    except Exception:
        if debug:
            raise
        img = _blank_image()
        graph = {}
        success = False

    return img, smiles, graph, success, style_config, mol_config

@func_set_timeout(5)  # Set timeout of 5 seconds
def generate_image_from_graph(graph,
                              style_config=None, mol_config=None, default_drawing_style=True,
                              debug=False):
    """
    Build molecule via RDKit from graph (to preserve solid/dashed wedges) and render using Indigo.
    This function is used for the cycle consistency check during training.
    
    Args:
        graph: Graph dict with 'symbols', 'edges', 'coords'
        style_config: dict, drawing style config for Indigo
        mol_config: dict, molecule config for Indigo
        debug: If True, raise exceptions instead of returning fallback
    
    Returns:
        img_pred: np.ndarray (H, W, 3) Color image
        style_config: dict, drawing style configuration used
        mol_config: dict, molecule configuration used
        success: bool, whether the generation was successful
    """
    try:
        mol = Chem.RWMol()
        symbols = graph['symbols']
        n = len(symbols)
        ids = []
        for i in range(n):
            symbol = symbols[i]
            if symbol[0] == '[':
                symbol = symbol[1:-1]
            if symbol in RGROUP_SYMBOLS:
                atom = Chem.Atom("*")
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    atom.SetIsotope(int(symbol[1:]))
                Chem.SetAtomAlias(atom, symbol)
            elif symbol in ABBREVIATIONS:
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)
            else:
                try:  # try to get SMILES of atom
                    atom = Chem.AtomFromSmiles(symbols[i])
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                except:  # otherwise, abbreviation or condensed formula
                    atom = Chem.Atom("*")
                    Chem.SetAtomAlias(atom, symbol)

            if atom.GetSymbol() == '*':
                atom.SetProp('molFileAlias', symbol)

            idx = mol.AddAtom(atom)
            assert idx == i
            ids.append(idx)

        edges = graph['edges']
        for i in range(n):
            for j in range(i + 1, n):
                if edges[i][j] == 1:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                elif edges[i][j] == 2:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
                elif edges[i][j] == 3:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
                elif edges[i][j] == 4:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
                elif edges[i][j] == 5:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

        coords = graph['coords']
        mol = _verify_chirality(mol, coords, edges, debug) # Atom coodinates are set here
        mol_block = Chem.MolToMolBlock(mol)

        indigo = Indigo()
        renderer = IndigoRenderer(indigo)

        mol = indigo.loadMolecule(mol_block)

        if style_config is None or mol_config is None: # Generate style config if not provided
            mol, style_config, mol_config = _generate_style_config(mol, default_drawing_style)
        _apply_style_config(indigo, style_config, mol, mol_config)

        mol.layout()
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # 1 flag for color
        success = True
    except Exception:
        if debug:
            raise
        img = _blank_image()
        success = False
    return img, style_config, mol_config, success
# --------------------------------------------------------------------------------------