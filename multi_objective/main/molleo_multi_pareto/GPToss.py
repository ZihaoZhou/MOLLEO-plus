import sys
import traceback
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import main.molleo_multi_pareto.crossover as co, main.molleo_multi_pareto.mutate as mu
import random
MINIMUM = 1e-10
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from openai import OpenAI
from dotenv import load_dotenv
import os

import networkx as nx
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

load_dotenv()

LLM_BASE_URL = os.getenv("MOLLEO_LLM_BASE_URL", "https://gpt-oss-120b-andrew.nrp-nautilus.io/v1")
LLM_API_KEY = os.getenv("OSS_KEY") or os.getenv("CLIENT_API_KEY")
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

from rdkit.Chem import GetPeriodicTable

pt = GetPeriodicTable()


ALKYL_FRAGMENTS = {
    "methyl": "[*:1]C",
    "ethyl": "[*:1]CC",
    "propyl": "[*:1]CCC",
    "isopropyl": "[*:1]C(C)C",
    "tert_butyl": "[*:1]C(C)(C)C",
    "cyclopropyl": "[*:1]C1CC1",
    "cyclobutyl": "[*:1]C1CCC1",
    "cyclopentyl": "[*:1]C1CCCC1",
    "cyclohexyl": "[*:1]C1CCCCC1"
}

HALOGEN_FRAGMENTS = {
    "fluoro": "[*:1]F",
    "chloro": "[*:1]Cl",
    "bromo": "[*:1]Br",
    "iodo": "[*:1]I"
}

HETEROATOM_FRAGMENTS = {
    "hydroxyl": "[*:1]O",
    "methoxy": "[*:1]OC",
    "ethoxy": "[*:1]OCC",
    "amine": "[*:1]N",
    "methylamine": "[*:1]NC",
    "dimethylamine": "[*:1]N(C)C",
    "thiol": "[*:1]S",
    "methylthio": "[*:1]SC"
}

CARBONYL_FRAGMENTS = {
    "aldehyde": "[*:1]C=O",
    "ketone_methyl": "[*:1]C(=O)C",
    "carboxylic_acid": "[*:1]C(=O)O",
    "ester_methyl": "[*:1]C(=O)OC",
    "amide": "[*:1]C(=O)N",
    "amide_methyl": "[*:1]C(=O)NC",
    "urea": "[*:1]NC(=O)N",
    "carbamate": "[*:1]OC(=O)N"
}

POLAR_FRAGMENTS = {
    "hydroxymethyl": "[*:1]CO",
    "aminoethyl": "[*:1]CCN",
    "dimethylaminoethyl": "[*:1]CCN(C)C",
    "morpholine": "[*:1]N1CCOCC1",
    "piperazine": "[*:1]N1CCNCC1",
    "piperidine": "[*:1]C1CCNCC1"
}

LARGE_RING_FRAGMENTS = {
    # Aromatics
    "phenyl": "[*:1]c1ccccc1",
    "benzyl": "[*:1]Cc1ccccc1",
    "phenoxy": "[*:1]Oc1ccccc1",
    "benzoyl": "[*:1]C(=O)c1ccccc1",
    
    # Heteroaromatics (Single Ring)
    "pyridyl_4": "[*:1]c1ccncc1",         # Attached at 4-position
    "pyridyl_3": "[*:1]c1cnccc1",         # Attached at 3-position
    "pyridyl_2": "[*:1]c1ncccc1",         # Attached at 2-position
    "pyrimidinyl": "[*:1]c1ncccn1",
    "thienyl": "[*:1]c1sccc1",
    "furanyl": "[*:1]c1occc1",
    
    # Fused Aromatics & Heterocycles
    "naphthyl_1": "[*:1]c1cccc2ccccc12",
    "naphthyl_2": "[*:1]c1ccc2ccccc2c1",
    "indole_3": "[*:1]c1c[nH]c2ccccc12",  # Common attachment point (tryptophan-like)
    "indole_N": "[*:1]n1ccc2ccccc12",     # Attached at Nitrogen
    "benzimidazole": "[*:1]c1nc2ccccc2[nH]1",
    "quinoline": "[*:1]c1cccc2ncccc12",
    "isoquinoline": "[*:1]c1cncc2ccccc12",
    "benzodioxole": "[*:1]c1ccc2OCOc2c1", # Often used to improve solubility/metabolic stability
    "benzofuran": "[*:1]c1oc2ccccc2c1",

    # Bulky/Bicyclic Aliphatics
    "adamantyl": "[*:1]C12CC3CC(CC(C3)C1)C2", # Very bulky lipophilic group
    "norbornyl": "[*:1]C1CC2CCC1C2",
    "biphenyl": "[*:1]c1ccc(cc1)c2ccccc2"
}

FRAGMENTS = {
    **ALKYL_FRAGMENTS,
    **HALOGEN_FRAGMENTS,
    **HETEROATOM_FRAGMENTS,
    **CARBONYL_FRAGMENTS,
    **POLAR_FRAGMENTS,
    **LARGE_RING_FRAGMENTS
}


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{smiles}: Invalid SMILES")
    Chem.SanitizeMol(mol)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def validate_smiles(smiles: str) -> dict:
    """Validates SMILES string and returns basic properties."""
    try:
        mol = mol_from_smiles(smiles)
        return {
            "is_valid": True,
            "canonical_smiles": canonical_smiles(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds()
        }
    except Exception as e:
        return {"is_valid": False, "error": str(e)}


def compute_atom_centralities(mol) -> dict:
    n = mol.GetNumAtoms()

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    betweenness = nx.betweenness_centrality(G, normalized=True)

    return betweenness

def get_ligand_structure(smiles: str) -> dict:
    try:
        mol = mol_from_smiles(smiles)
        centralities = compute_atom_centralities(mol)
        pts = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = atom.GetNeighbors()
            pts.append({
                "atom_index": idx,
                "element": atom.GetSymbol(),
                "substitutable_hydrogens": atom.GetTotalNumHs(),
                "available_valences": pt.GetDefaultValence(atom.GetAtomicNum()) - atom.GetTotalValence(),
                "num_neighbors": atom.GetDegree(),
                "neighbor_indices": [n.GetIdx() for n in neighbors],
                "is_in_ring": atom.IsInRing(),
                "centrality": round(float(centralities[idx]), 2),
            })
        return {"structure_information": pts}
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_atom(
    smiles: str,
    target_atom_index: int,
    new_atom: str,
    bond_type: str = "SINGLE"
) -> dict:
    """
    Adds a single atom to the molecule.
    """
    try:
        mol = mol_from_smiles(smiles)
        
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False, 
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens for substitution"
            }
        
        rw = Chem.RWMol(mol)

        bond_map = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE
        }

        if bond_type not in bond_map:
            return {"success": False, "error": f"Invalid bond type: {bond_type}. Use SINGLE, DOUBLE, or TRIPLE"}

        new_idx = rw.AddAtom(Chem.Atom(new_atom))
        rw.AddBond(target_atom_index, new_idx, bond_map[bond_type])
        
        target = rw.GetAtomWithIdx(target_atom_index)
        target.SetNumExplicitHs(max(0, target.GetNumExplicitHs() - 1))

        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
def replace_atom(
smiles: str,
atom_index: int,
new_element: str
) -> dict:
    """
    Replaces an atom with a different element.
    """
    try:
        mol = mol_from_smiles(smiles)
        
        # Validate atom index
        if atom_index < 0 or atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {atom_index}"}
        
        rw = Chem.RWMol(mol)
        old_atom = rw.GetAtomWithIdx(atom_index)
        old_element = old_atom.GetSymbol()
        
        try:
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_element)
        except:
            return {"success": False, "error": f"Invalid element symbol: {new_element}"}

        old_atom.SetAtomicNum(atomic_num)
        
        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw),
            "old_element": old_element,
            "new_element": new_element,
            "warning": "Atom replacement may create invalid valence states. Always validate result."
        }
    except Chem.AtomValenceException as e:
        return {
            "success": False, 
            "error": f"Valence error: {new_element} cannot have the same bonding pattern as the original atom. {str(e)}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_functional_group(
    smiles: str,
    target_atom_index: int,
    group: str,
    bond_type: str = "SINGLE"
) -> dict:
    """
    Adds a functional group by replacing a hydrogen.
    """
    try:
        if group not in FRAGMENTS:
            available = ", ".join(sorted(FRAGMENTS.keys()))
            return {"success": False, "error": f"Unknown functional group: {group}. Available: {available}"}

        mol = mol_from_smiles(smiles)
        
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False,
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens for substitution"
            }

        frag = Chem.MolFromSmiles(FRAGMENTS[group])
        if frag is None:
            return {"success": False, "error": f"Invalid fragment SMILES for group: {group}"}

        dummy_atoms = [
            atom for atom in frag.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        if len(dummy_atoms) != 1:
            return {"success": False, "error": "Fragment must contain exactly one [*:1] dummy atom"}

        dummy_atom = dummy_atoms[0]
        dummy_idx = dummy_atom.GetIdx()

        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            return {"success": False, "error": "Dummy atom must have exactly one neighbor"}

        attach_idx_frag = neighbors[0].GetIdx()

        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)

        mol_n_atoms = mol.GetNumAtoms()
        dummy_idx_combo = mol_n_atoms + dummy_idx
        attach_idx_combo = mol_n_atoms + attach_idx_frag

        bond_map = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE
        }

        rw.AddBond(
            target_atom_index,
            attach_idx_combo,
            bond_map[bond_type]
        )

        rw.RemoveAtom(dummy_idx_combo)

        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw),
            "group_added": group
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_substructure(
    smiles: str,
    target_atom_index: int,
    fragment_string: str,
    bond_type: str = "SINGLE"
) -> dict:
    """
    Attaches a custom structure to a scaffold at a specific atom index.
    
    Args:
        smiles: The SMILES string of the scaffold molecule.
        target_atom_index: The index of the atom on the scaffold to attach to.
        fragment_string: A SMILES/SMARTS string of the group to add. 
                         MUST contain exactly one attachment point labeled '[*:1]'.
                         Example: "[*:1]C1CC1" (Cyclopropyl)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "Invalid scaffold SMILES provided."}
        
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False,
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens to replace."
            }

        frag = Chem.MolFromSmiles(fragment_string)
        if frag is None:
            frag = Chem.MolFromSmarts(fragment_string)
            if frag is None:
                return {"success": False, "error": f"Could not parse fragment structure: {fragment_string}"}

        dummy_atoms = [
            atom for atom in frag.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        
        if len(dummy_atoms) != 1:
            return {
                "success": False, 
                "error": "Fragment must contain exactly one attachment point labeled '[*:1]'"
            }

        dummy_atom = dummy_atoms[0]
        dummy_idx = dummy_atom.GetIdx()

        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            return {"success": False, "error": "Attachment point [*:1] must have exactly one neighbor"}

        attach_idx_frag = neighbors[0].GetIdx()

        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)

        mol_n_atoms = mol.GetNumAtoms()
        dummy_idx_combo = mol_n_atoms + dummy_idx
        attach_idx_combo = mol_n_atoms + attach_idx_frag

        bond_map = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE
        }
        rw.AddBond(
            target_atom_index,
            attach_idx_combo,
            bond_map[bond_type]
        )
        rw.RemoveAtom(dummy_idx_combo)

        try:
            Chem.SanitizeMol(rw)
        except ValueError as e:
            return {
                "success": False, 
                "error": f"Chemical sanitization failed (likely valence error): {str(e)}"
            }

        return {
            "success": True,
            "new_smiles": Chem.MolToSmiles(rw),
            "structure_added": fragment_string
        }
        
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def replace_substructure(
    smiles: str,
    query_smarts: str,
    replacement_smiles: str,
    anchor_atom_idx: int,
    replace_all: bool = False,
) -> dict:
    try:
        mol = mol_from_smiles(smiles)
        if mol is None:
            return {"success": False, "error": "Invalid scaffold SMILES provided."}

        # ── Validate query ────────────────────────────────────────────────
        query = Chem.MolFromSmarts(query_smarts.replace("[*:1]", ""))
        if query is None:
            return {"success": False, "error": f"Could not parse query SMARTS: {query_smarts}"}

        # ── Validate replacement ──────────────────────────────────────────
        if "[*:1]" not in replacement_smiles:
            replacement_smiles = "[*:1]"+replacement_smiles
            
        replacement = Chem.MolFromSmiles(replacement_smiles)
        if replacement is None:
            replacement = Chem.MolFromSmarts(replacement_smiles)
            if replacement is None:
                return {"success": False, "error": f"Could not parse replacement structure: {replacement_smiles}"}

        repl_dummies = [
            atom for atom in replacement.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        if len(repl_dummies) != 1:
            return {
                "success": False,
                "error": (
                    f"replacement_smiles must contain exactly one '[*:1]' marking the "
                    f"attachment point (found {len(repl_dummies)}). "
                    f"Example: '[*:1]C1CC1' for cyclopropyl."
                ),
            }

        if len(list(repl_dummies[0].GetNeighbors())) != 1:
            return {
                "success": False,
                "error": "Attachment point '[*:1]' in replacement_smiles must have exactly one neighbor.",
            }

        # ── Match & replace ───────────────────────────────────────────────
        matches = mol.GetSubstructMatches(query)
        if not matches:
            return {"success": False, "error": "No matching substructure found in the molecule."}

        if anchor_atom_idx is not None:
            if anchor_atom_idx < 0 or anchor_atom_idx >= mol.GetNumAtoms():
                return {
                    "success": False,
                    "error": f"anchor_atom_idx {anchor_atom_idx} is out of range for a molecule with {mol.GetNumAtoms()} atoms.",
                }
            filtered = [m for m in matches if anchor_atom_idx in m]
            if not filtered:
                return {
                    "success": False,
                    "error": (
                        f"anchor_atom_idx {anchor_atom_idx} is not part of any substructure match. "
                        f"Found {len(matches)} match(es) not containing this atom."
                    ),
                }
            matches = filtered

        result_mol = mol
        match_set = set(matches[0])

        
        # Find all scaffold atoms neighbouring the match (the cut bonds)
        scaffold_attach_indices = []
        for atom_idx in match_set:
            for neighbor in result_mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                if neighbor.GetIdx() not in match_set:
                    scaffold_attach_indices.append(neighbor.GetIdx())
        if len(scaffold_attach_indices) != 1:
            return {
                "success": False,
                "error": (
                    f"Matched substructure has {len(scaffold_attach_indices)} bond(s) "
                    "to the rest of the scaffold; expected exactly 1. "
                    "Refine your query_smarts to select a terminal substructure."
                ),
            }

        scaffold_attach_idx = scaffold_attach_indices[0]


        repl_dummy = repl_dummies[0]
        repl_dummy_idx = repl_dummy.GetIdx()
        repl_attach_idx = list(repl_dummy.GetNeighbors())[0].GetIdx()

        n_scaffold_atoms = result_mol.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(result_mol, replacement))

        repl_dummy_combo = n_scaffold_atoms + repl_dummy_idx
        repl_attach_combo = n_scaffold_atoms + repl_attach_idx

        rw.AddBond(scaffold_attach_idx, repl_attach_combo, Chem.BondType.SINGLE)

        # Remove matched atoms + replacement dummy in reverse index order
        # to keep earlier indices valid during deletion
        for idx in sorted(match_set | {repl_dummy_combo}, reverse=True):
            rw.RemoveAtom(idx)

        result_mol = rw.GetMol()

        # ── Connectivity check ────────────────────────────────────────────
        frags = Chem.GetMolFrags(result_mol, asMols=True)
        if len(frags) > 1:
            return {
                "success": False,
                "error": (
                    "Replacement produced disconnected fragments. "
                    "Check that '[*:1]' in query_smarts is on the correct attachment bond."
                ),
            }

        # ── Sanitize ──────────────────────────────────────────────────────
        try:
            Chem.SanitizeMol(result_mol)
        except ValueError as e:
            return {
                "success": False,
                "error": f"Chemical sanitization failed (likely valence error): {str(e)}",
            }

        return {
            "success": True,
            "new_smiles": canonical_smiles(result_mol),
            "num_matches": len(matches),
            "replaced_all": replace_all,
            "warning": "Only first match replaced" if not replace_all and len(matches) > 1 else None,
        }

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def remove_substructure(
    smiles: str,
    substructure_smarts: str,
    anchor_atom_idx: int
) -> dict:
    """
    Removes all instances of a substructure match from the molecule.
    Fails if the removal causes the molecule to break into disconnected fragments.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        query = Chem.MolFromSmarts(substructure_smarts)

        if not mol:
            return {"success": False, "error": "Invalid base SMILES"}
        if not query:
            return {"success": False, "error": "Invalid SMARTS pattern"}

        matches = mol.GetSubstructMatches(query)
        
        if not matches:
            return {"success": False, "error": "Substructure not found in molecule"}
        
        if anchor_atom_idx is not None:
            if anchor_atom_idx < 0 or anchor_atom_idx >= mol.GetNumAtoms():
                return {
                    "success": False,
                    "error": f"anchor_atom_idx {anchor_atom_idx} is out of range for a molecule with {mol.GetNumAtoms()} atoms.",
                }
            filtered = [m for m in matches if anchor_atom_idx in m]
            if not filtered:
                return {
                    "success": False,
                    "error": (
                        f"anchor_atom_idx {anchor_atom_idx} is not part of any substructure match. "
                        f"Found {len(matches)} match(es) not containing this atom."
                    ),
                }
            matches = filtered
            
        indices_to_remove = set()
        for match in matches:
            indices_to_remove.update(match)
            
        rw = Chem.RWMol(mol)
        
        sorted_indices = sorted(list(indices_to_remove), reverse=True)
        
        for idx in sorted_indices:
            rw.RemoveAtom(idx)

        frags = Chem.GetMolFrags(rw, asMols=True)
        
        if len(frags) > 1:
            return {
                "success": False, 
                "error": "Operation failed: Removal resulted in disconnected fragments (e.g., broken linker)."
            }
            
        final_mol = rw

        Chem.SanitizeMol(final_mol)
        return {
            "success": True,
            "new_smiles": Chem.MolToSmiles(final_mol, isomericSmiles=True),
            "removed_count": len(sorted_indices)
        }

    except Exception as e:
        return {"success": False, "error": f"Removal failed: {str(e)}"}
    

def _fragment_molecule(smiles: str, cut_atom_idx: int) -> dict:
    """
    Splits a molecule at the first non-ring bond from the specified atom.
    Returns head (containing cut_atom_idx) and tail as RDKit mols,
    each with exactly one [*:1] marking the cut point.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {"success": False, "error": f"Invalid SMILES: {smiles}"}

    n_atoms = mol.GetNumAtoms()
    if cut_atom_idx < 0 or cut_atom_idx >= n_atoms:
        return {
            "success": False,
            "error": f"Invalid atom index {cut_atom_idx} for molecule with {n_atoms} atoms.",
        }

    atom = mol.GetAtomWithIdx(cut_atom_idx)

    # Find the first non-ring bond from this atom — ring bonds can't produce 2 clean fragments
    cut_bond_idx = None
    for neighbor in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(cut_atom_idx, neighbor.GetIdx())
        if not bond.IsInRing():
            cut_bond_idx = bond.GetIdx()
            break

    if cut_bond_idx is None:
        return {
            "success": False,
            "error": (
                f"Atom {cut_atom_idx} ({atom.GetSymbol()}) has no non-ring bonds. "
                "Choose an atom with at least one acyclic bond to the scaffold."
            ),
        }

    frag_mol = Chem.FragmentOnBonds(mol, [cut_bond_idx], addDummies=True)
    frag_idx_sets = Chem.GetMolFrags(frag_mol)           # atom indices in frag_mol coords
    frag_mols    = Chem.GetMolFrags(frag_mol, asMols=True)

    if len(frag_mols) != 2:
        return {
            "success": False,
            "error": (
                f"Expected 2 fragments after cutting atom {cut_atom_idx}, "
                f"got {len(frag_mols)}. The atom may be fully embedded in a ring system."
            ),
        }

    def _normalize_dummy(mol_raw: Chem.Mol) -> Chem.Mol:
        """Set the dummy atom's map number to 1 and clear any isotope label."""
        rw = Chem.RWMol(mol_raw)
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0:   # wildcard / dummy
                a.SetAtomMapNum(1)
                a.SetIsotope(0)
        return rw.GetMol()

    head_mol, tail_mol = None, None
    for idx_set, frag in zip(frag_idx_sets, frag_mols):
        if cut_atom_idx in idx_set:
            head_mol = _normalize_dummy(frag)
        else:
            tail_mol = _normalize_dummy(frag)

    if head_mol is None or tail_mol is None:
        return {"success": False, "error": "Internal error: could not identify head/tail fragments."}

    return {"success": True, "head": head_mol, "tail": tail_mol}


def _join_fragments(frag_a: Chem.Mol, frag_b: Chem.Mol) -> Chem.Mol | None:
    """
    Joins two fragments at their [*:1] attachment points.
    Uses the same CombineMols → AddBond → RemoveAtom pattern as add_substructure.
    Returns the sanitized product mol, or None if anything fails.
    """
    def _get_attachment(mol: Chem.Mol) -> tuple[int, int] | tuple[None, None]:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == 1:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) == 1:
                    return atom.GetIdx(), neighbors[0].GetIdx()
        return None, None

    dummy_a, attach_a = _get_attachment(frag_a)
    dummy_b, attach_b = _get_attachment(frag_b)

    if dummy_a is None or dummy_b is None:
        return None

    n_a = frag_a.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(frag_a, frag_b))

    attach_b_combo = n_a + attach_b
    dummy_b_combo  = n_a + dummy_b

    rw.AddBond(attach_a, attach_b_combo, Chem.BondType.SINGLE)

    # Remove both dummies highest-index first to keep earlier indices stable
    for idx in sorted([dummy_a, dummy_b_combo], reverse=True):
        rw.RemoveAtom(idx)

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def crossover_molecules(
    smiles_a: str,
    cut_idx_a: int,
    smiles_b: str,
    cut_idx_b: int,
) -> dict:
    """
    Splits two molecules at specified atoms and recombines their fragments.
    Tries all 4 combinations (head+head, head+tail, tail+head, tail+tail)
    and returns the first chemically valid result.

    Each molecule is cleaved at the first non-ring bond from the specified atom.
    The fragment that retains the cut atom is the 'head'; the other is the 'tail'.

    Args:
        smiles_a:   SMILES of the first molecule.
        cut_idx_a:  Atom index in molecule A at which to cut.
        smiles_b:   SMILES of the second molecule.
        cut_idx_b:  Atom index in molecule B at which to cut.
    """
    # ── Fragment both molecules ───────────────────────────────────────────
    split_a = _fragment_molecule(smiles_a, cut_idx_a)
    if not split_a["success"]:
        return {"success": False, "error": f"Could not fragment molecule A: {split_a['error']}"}

    split_b = _fragment_molecule(smiles_b, cut_idx_b)
    if not split_b["success"]:
        return {"success": False, "error": f"Could not fragment molecule B: {split_b['error']}"}

    head_a, tail_a = split_a["head"], split_a["tail"]
    head_b, tail_b = split_b["head"], split_b["tail"]

    # ── Try all 4 recombinations ──────────────────────────────────────────
    combinations = [
        ("head_a + head_b", head_a, head_b),
        ("head_a + tail_b", head_a, tail_b),
        ("tail_a + head_b", tail_a, head_b),
        ("tail_a + tail_b", tail_a, tail_b),
    ]

    all_results = []
    first_valid = None
    all_valid = []
    for label, frag1, frag2 in combinations:
        product = _join_fragments(frag1, frag2)
        if product is not None:
            smi = canonical_smiles(product)
            all_results.append({"combination": label, "smiles": smi, "valid": True})
            all_valid.append({"combination": label, "smiles": smi, "valid": True})
            if first_valid is None:
                first_valid = {"combination": label, "smiles": smi}
        else:
            all_results.append({"combination": label, "smiles": None, "valid": False})

    if first_valid is None:
        return {
            "success": False,
            "error": "All 4 fragment combinations failed sanitization.",
            "all_results": all_results,
        }
    choice = np.random.choice(all_valid)
    return {
        "success": True,
        "new_smiles": choice["smiles"],
        "combination_used": choice["combination"],
    }


def calculate_properties(smiles: str) -> dict:
    """Calculates common molecular descriptors."""
    try:
        mol = mol_from_smiles(smiles)
        return {
            "qed": round(QED.qed(mol), 2),
            "sa": round(sascorer.calculateScore(mol), 2),
            "mw": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 2),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------
# Tool registry
# ------------------------------------------------------------
TOOLS = {
    "add_atom": add_atom,
    "replace_atom": replace_atom,
    "add_functional_group": add_functional_group,
    "add_substructure": add_substructure,
    "replace_substructure": replace_substructure,
    "remove_substructure": remove_substructure,
    "crossover_molecules": crossover_molecules
}


# ------------------------------------------------------------
# Tool schemas (OpenAI Responses API)
# ------------------------------------------------------------
TOOL_SCHEMAS = [
       {
    "type": "function",
    "name": "crossover_molecules",
    "description": "Splits two molecules at the specified indices and recombines the resulting fragments. Do NOT split molecules at a ring index. Try to split each molecule roughly in half, use high centrality as measure for good splitting index.",
    "parameters": {
        "type": "object",
        "properties": {
            "smiles_a": {
                "type": "string",
                "description": "SMILES string of the first molecule."
            },
            "cut_idx_a": {
                "type": "integer",
                "description": "Index of the atom in Molecule 1 where the split should occur."
            },
            "smiles_b": {
                "type": "string",
                "description": "SMILES string of the second molecule."
            },
            "cut_idx_b": {
                "type": "integer",
                "description": "Index of the atom in Molecule 2 where the split should occur."
            },
        },
        "required": [
            "smiles_a",
            "cut_idx_a",
            "smiles_b",
            "cut_idx_b"
        ]
    }
},
    {
        "type": "function",
        "name": "add_atom",
        "description": "Attach new atom to the specified index",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "new_atom": {"type": "string"},
                "bond_type": {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE"]},
            },
            "required": ["smiles", "target_atom_index", "new_atom"],
        },
    },
    {
        "type": "function",
        "name": "replace_atom",
        "description": "Replace atom at specified index",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "atom_index": {"type": "integer"},
                "new_element": {"type": "string"},
            },
            "required": ["smiles", "atom_index", "new_element"],
        },
    },
    {
        "type": "function",
        "name": "add_functional_group",
        "description": "Attach specified functional group (from provided enum) at the specified index. For names in the enum with a suffix, _N means to attach that group via nitrogen, while _# for any number indicates the attachment position on the ring.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "group": {
                    "type": "string",
                    "enum": [
                        "methyl","ethyl","propyl","isopropyl","tert_butyl",
                        "cyclopropyl","cyclobutyl","cyclopentyl","cyclohexyl",
                        "fluoro","chloro","bromo","iodo",
                        "hydroxyl","methoxy","ethoxy",
                        "amine","methylamine","dimethylamine",
                        "thiol","methylthio",
                        "aldehyde","ketone_methyl","carboxylic_acid",
                        "ester_methyl","amide","amide_methyl","urea","carbamate",
                        "hydroxymethyl","aminoethyl","dimethylaminoethyl",
                        "morpholine","piperazine","piperidine",
                        "phenyl", "benzyl", "phenoxy", "benzoyl",
                        "pyridyl_4", "pyridyl_3", "pyridyl_2", "pyrimidinyl",
                        "thienyl", "furanyl", "naphthyl_1", "naphthyl_2",
                        "indole_3", "indole_N", "benzimidazole", "quinoline",
                        "isoquinoline", "benzodioxole", "benzofuran",
                        "adamantyl", "norbornyl", "biphenyl"
                        
                    ],
                },
                "bond_type": {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE"]},
            },
            "required": ["smiles", "target_atom_index", "group", "bond_type"],
        },
    },
    {
        "type": "function",
        "name": "add_substructure",
        "description": "Attaches a custom SMILES group or structure to a specific atom index.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                },
                "target_atom_index": {
                    "type": "integer",
                },
                "fragment_string": {
                    "type": "string",
                    "description": "MUST contain exactly one attachment point labeled '[*:1]'. Example: '[*:1]C1CC1' for cyclopropyl."
                },
                "bond_type": {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE"]},

            },
            "required": ["smiles", "target_atom_index", "fragment_string"]
        }
    },
    {
        "type": "function",
        "name": "replace_substructure",
        "description": "Replace terminal SMARTS-specified substructure with custom SMILES; only replaces first SMARTS match. Try to target terminal substructures.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "query_smarts": {
                    "type": "string",
                    "description": "IMPORTANT: Do NOT include any labeled attachment points in this SMARTS argument."
                },
                "replacement_smiles": {
                    "type": "string",
                    "description": "MUST contain exactly one labeled attachment point '[*:1]'"
                },
                "anchor_atom_idx":{
                    "type": "integer",
                    "description": "Index of any atom within your desired substructure to replace. Eliminates ambiguity when matching your SMARTS."
                }
            },
            "required": ["smiles", "query_smarts", "replacement_smiles", "anchor_atom_idx"],
        },
    },
    {
        "type": "function",
        "name": "remove_substructure",
        "description": "Delete SMARTS-specified substructure. Try to target terminal substructures.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "substructure_smarts": {"type": "string"},
                "anchor_atom_idx":{
                    "type": "integer",
                    "description": "Index of any atom within your desired substructure to remove. Eliminates ambiguity when matching your SMARTS."
                }
            },
            "required": ["smiles", "substructure_smarts", "anchor_atom_idx"],
        },
    },
]


# ------------------------------------------------------------
# Agent state
# ------------------------------------------------------------
@dataclass
class AgentState:
    current_smiles: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None


# ------------------------------------------------------------
# Tool executor
# ------------------------------------------------------------
def execute_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"success": False, "error": "Unknown tool"}
    return TOOLS[name](**args)


# ------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------
def run_agent(
    initial_smiles: list[str],
    prompt: str,
    max_steps: int = 10,
    index: int = 0
) -> AgentState:

    state = AgentState(
        current_smiles=""
    )

    system_context = (
        "You are a molecular design agent.\n"
        "You may ONLY modify molecules using tools.\n"
        "Only make one modification at a time.\n"
        "Read the parameter descriptions for the tools very carefully.\n"
        "Always ensure that your modifications don't break valence rules and does not result in a fragmented molecule."
    )
    
    # no crossovers (1 molecule)
    # user_goal = (
    #     f"Goal: {prompt}\n"
    #     f"Initial SMILES: {initial_smiles[0]}"
    # )
    
    # messages = [
    #     {"role": "system", "content": system_context},
    #     {"role": "user", "content": user_goal},
    #     {"role": "user", "content": f"Possible attachment points: {str(get_attachment_points(initial_smiles[0]))}"},
    #     {"role": "user", "content": f"Molecule properties: {str(calculate_properties(initial_smiles[0]))}"}
    # ]
    
    # crossovers (2 molecules)
    user_goal = (
        f"Goal: {prompt}\n"
        # f"Initial SMILES:\n1. {initial_smiles[0]}\n2. {initial_smiles[1]}"
    )    

    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_goal},
        {"role": "user", "content": f"Ligand structure and possible attachment points for ligand 1: {str(get_ligand_structure(initial_smiles[0]))}\nLigand structure and possible attachment points for ligand 2: {str(get_ligand_structure(initial_smiles[1]))}"},
        {"role": "user", "content": f"Molecule properties for ligand 1: {str(calculate_properties(initial_smiles[0]))}\nMolecule properties for ligand 2: {str(calculate_properties(initial_smiles[1]))}"}
    ]
    
    modification_tools = ["add_atom", "replace_atom", "add_functional_group", "add_substructure", "remove_substructure", "replace_substructure", "crossover_molecules"]

    should_break = False
    for step in range(max_steps):
        # print(f"\n[Index {str(index)}] CURRENT MESSAGES: {str(messages)}\n", flush=True)
        response = client.responses.create(
            model='gpt-oss-120b',
            input=messages,
            tools=TOOL_SCHEMAS
        )
        
        for msg in response.output:
            if msg.type == "reasoning":
                print(f"\n[Index {str(index)}] [Reasoning]: {msg.content}") 
            elif msg.type == "function_call":
                print(f"[Index {str(index)}] [Tool Call]: {msg}")
                
                messages.append(msg)
                tool_name = msg.name
                args = json.loads(msg.arguments)

                # if "smiles" in args:
                #     args["smiles"] = state.current_smiles

                result = execute_tool(tool_name, args)
                
                print(f"[Index {str(index)}] [Tool Result]: {result}")
                
                if "new_smiles" in result:
                    state.current_smiles = result["new_smiles"]
                
                state.history.append({
                    "step": step,
                    "tool": tool_name,
                    "arguments": args,
                    "result": result
                })
                
                messages.append({
                    "type": "function_call_output",
                    "call_id": msg.call_id,
                    "output": json.dumps(result)
                })
                if state.current_smiles:
                    if tool_name in modification_tools:
                        for i, item in enumerate(messages):
                            if isinstance(item, dict) and "content" in item and "possible attachment points" in item["content"]:
                                messages.pop(i)
                                break
                        
                        messages.append({
                            "role": "user",
                            "content": (
                                "Output FINAL_ANSWER if you have made sufficient modifications (make at most 3). Ensure that desired properties are maintained.\n"
                                f"Current SMILES: {state.current_smiles}\n"
                                f"Ligand structure and possible attachment points: {str(get_ligand_structure(state.current_smiles))}"
                            )
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Molecule properties: {str(calculate_properties(state.current_smiles))}"
                        })
                    
            else:
                content = msg.content[0].text
                print(f"[Index {str(index)}] [Message]: {content}")
                state.final_answer = content
                should_break = True
                break
        if should_break:
            break
    return state

def query_LLM(messages, index):
    response = client.responses.create(
        model='gpt-oss-120b',
        input=messages,
    )
        
    for msg in response.output:
        if msg.type == "reasoning":
            print(f"[Index {str(index)}] [Reasoning]: {msg.content}") 
        else:
            content = msg.content[0].text
            print(f"[Index {str(index)}] [Message]: {content}")
            return content

class GPToss:
    def __init__(self):
        self.task2description = {
                'c-met': 'I have two molecules and their docking scores to c-MET. The docking score measures how well a molecule binds to c-MET. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n',
                'brd4': 'I have two molecules and their docking scores to BRD4. The docking score measures how well a molecule binds to BRD4. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n',
                'qed': 'I have two molecules and their QED scores. The QED score measures the drug-likeness of the molecule.\n\n',
                'jnk3': 'I have two molecules and their JNK3 scores. The JNK3 score measures a molecular\'s biological activity against JNK3.\n\n',
                'drd2': 'I have two molecules and their DRD2 scores. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).\n\n',
                'gsk3b': 'I have two molecules and their GSK3$\beta$ scores. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'I have two molecules and their isomer scores. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'I have two molecules and their perindopril multiproperty objective scores. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings.\n\n',
                'sitagliptin_mpo': 'I have two molecules and their sitagliptin multiproperty objective scores. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O.\n\n',
                'ranolazine_mpo': 'I have two molecules and their ranolazine multiproperty objective scores. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms.\n\n',
                'thiothixene_rediscovery': 'I have two molecules and their thiothixene rediscovery measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered.\n\n',
                'mestranol_similarity': 'I have two molecules and their mestranol similarity scores. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol.\n\n',
                }
        self.task2objective = {
                'c-met': 'Please propose a new molecule that binds better to c-MET. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'brd4': 'Please propose a new molecule that binds better to BRD4. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        self.requirements = """\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your proposed molecule.\n3. The molecule should be valid.
        """
        self.task=None
        self.error_count = 0
        self.current_summary = ""

    def edit(self, idx, mating_tuples, mutation_rate, min_evaluators, max_evaluators, boltz_cache, single_parent=False, use_tools=False):
        task = self.task
        all_objectives = min_evaluators + max_evaluators
        obj_map = {"qed": "QED", "sa": "SA (Synthetic Accessibility)", "c-met": "Binding Affinity against the kinase c-MET", "brd4": "Binding Affinity against BRD4"}
        
        task_definition = "I have a molecule and its "
        for objective in all_objectives[:-1]:
            task_definition += obj_map[objective[0]]
            task_definition += ", "
        task_definition += "and "
        task_definition += obj_map[all_objectives[-1][0]]
        task_definition += ".\n"
        
        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_smiles = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        # print(parent_smiles)
        try: 
            task_objective = "\nI want to "
            for min_obj in min_evaluators:
                if min_obj[0] == "c-met" or min_obj[0] == "brd4":
                    task_objective += "improve "
                else:
                    task_objective += "minimize "
                task_objective += obj_map[min_obj[0]]
                task_objective += ", "
            for max_obj in max_evaluators[:-1]:
                if max_obj[0] == "c-met" or max_obj[0] == "brd4":
                    task_objective += "worsen "
                else:
                    task_objective += "maximize "
                task_objective += obj_map[max_obj[0]]
                task_objective += ", "
            task_objective += "and maximize " if max_evaluators[-1][0] != "c-met" and max_evaluators[-1][0] != "brd4" else "and worsen "
            task_objective += obj_map[max_evaluators[-1][0]]
            task_objective += ". "
            task_objective += "Recall that a more negative binding affinity is better, and a more positive binding affinity is worse. Please propose a new molecule better than the current molecule."
            
            if use_tools:
                #no crossovers (1 input molecule)
                # objective=f"{task_objective} Only make a few (at most 3) modifications, then respond with FINAL_ANSWER. Do not let molecular weight exceed 700.\n"

                #crossovers (2 input molecules)
                objective=f"{task_objective} I have given you two candidate ligands. Please propose a new molecule that binds better to c-MET. You are encouraged to make a crossover between the molecules on the first step, then mutate the resulting molecule. Only make a few modifications (at most 3), then respond with FINAL_ANSWER. Do not let molecular weight exceed 700.\n"
                mol_tuple = ''
                for i in range(len(parent)):
                    smiles = parent_smiles[i]
                    tu = f'\n{str(i+1)}. {smiles}\n'
                    for obj in all_objectives: 
                        tu += obj_map[obj[0]]
                        tu+= ": "
                        if obj[0] == "c-met" or obj[0] == "brd4":
                            score = round(boltz_cache[obj[0]][smiles], 2)
                        else:    
                            mol = Chem.MolFromSmiles(smiles)
                            score = round(obj[1](mol), 2)
                        tu += str(score)
                        tu += "\n"
                    mol_tuple = mol_tuple + tu
                objective += mol_tuple
                for i in range(10):
                    try:
                        # state = run_agent(parent_smiles[0], task_objective, max_steps=20, index=idx)
                        state = run_agent(
                            initial_smiles=parent_smiles,
                            prompt=objective,
                            max_steps=10,
                            index=idx
                        )
                        proposed_smiles = state.current_smiles
                        break
                    except Exception as e:
                        print(e)
                        continue
            else:
                #original prompt
                mol_tuple = ''
                for i in range(2):
                    tu = '\n[' + parent_smiles[i] + ',' + str(parent_scores[i]) + ']'
                    mol_tuple = mol_tuple + tu
                protein = ""
                task_objective = "\nI want to "
                for min_obj in min_evaluators:
                    if min_obj[0] == "c-met" or min_obj[0] == "brd4":
                        task_objective += "improve "
                        protein = min_obj[0]
                    else:
                        task_objective += "minimize "
                    task_objective += obj_map[min_obj[0]]
                    task_objective += ", "
                for max_obj in max_evaluators[:-1]:
                    if max_obj[0] == "c-met" or max_obj[0] == "brd4":
                        task_objective += "worsen "
                    else:
                        task_objective += "maximize "
                    task_objective += obj_map[max_obj[0]]
                    task_objective += ", "
                task_objective += "and maximize " if max_evaluators[-1][0] != "c-met" and max_evaluators[-1][0] != "brd4" else "and worsen "
                task_objective += obj_map[max_evaluators[-1][0]]
                task_objective += ". "
                task_objective += self.task2objective[protein]
                prompt = self.task2description[protein] + mol_tuple + task_objective + self.requirements
            
                # modified prompt
                # mol_tuple = ''
                # for i in range(len(parent)):
                #     smiles = parent_smiles[i]
                #     tu = f'\n{str(i+1)}. {smiles}\n'
                #     for obj in all_objectives: 
                #         tu += obj_map[obj[0]]
                #         tu+= ": "
                #         if obj[0] == "c-met" or obj[0] == "brd4":
                #             score = round(boltz_cache[obj[0]][smiles], 2)
                #         else:    
                #             mol = Chem.MolFromSmiles(smiles)
                #             score = round(obj[1](mol), 2)
                #         tu += str(score)
                #         tu += "\n"
                #     mol_tuple = mol_tuple + tu
                # prompt = task_definition + mol_tuple + task_objective + self.requirements
                    
                print(f"[Index {str(idx)}] Prompt: " + prompt, flush=True)
                messages = [{"role": "user", "content": prompt}]
                for i in range(10):
                    try:
                        r = query_LLM(messages, idx)
                        break
                    except Exception as e:
                        print(e)
                        continue
                proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
                proposed_smiles = proposed_smiles.replace('"', '')
           
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(f"LLM-GENERATED: {proposed_smiles}", flush=True)
            assert proposed_smiles != None
            # score = self.oracle(proposed_smiles)
            
            # messages.append({"role": "assistant", "content": r})
            # summary_prompt = "Software shows that the ligand you generated ("+proposed_smiles+") had a binding affinity of "+str(-score)+" kcal/mol.\n"
            # summary_prompt += f"Update our current accumulated knowledge about {protein}. Briefly summarize the most important details and information about the binding target that we've learned so far. Remember that a more negative binding affinity is better. Disregard the previous output format, and do NOT generate any new molecules at this time."
            # messages.append({"role": "user", "content": summary_prompt})
            # summary = query_LLM(messages)
            # self.current_summary = summary.replace("assistant\n\n", "")

            return proposed_smiles
        except Exception as e:
            traceback.print_exc()
            print(f"{type(e).__name__} {e}")
            self.error_count += 1
            print("NUM LLM ERRORS: " + str(self.error_count), flush=True)
            score = 0
            new_child = co.crossover(Chem.MolFromSmiles(parent_smiles[0]), Chem.MolFromSmiles(parent_smiles[1]))
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            if new_child is not None: 
                smiles = Chem.MolToSmiles(new_child, isomericSmiles=False, canonical=True)
                print(f"NON-LLM GENERATED: {smiles}")
                
                # messages.append({"role": "assistant", "content": smiles})
                # summary_prompt = "Software shows that the ligand "+smiles+" has a binding affinity of "+str(-score)+" kcal/mol.\n"
                # summary_prompt += f"Based on this docking result, update our current accumulated knowledge about {protein}. Briefly summarize the most important details and information about the binding target that we've learned so far. Remember that a more negative binding affinity is better. Disregard the previous output format, and do NOT generate any new molecules at this time."
                # messages.append({"role": "user", "content": summary_prompt})
                # summary = query_LLM(messages)
                # self.current_summary = summary.replace("assistant\n\n", "")
            return smiles

def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return smi_canon
    except:
        return None
