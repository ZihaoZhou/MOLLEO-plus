import sys
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import main.molleo.crossover as co, main.molleo.mutate as mu
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

from rdkit.Chem import QED
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

load_dotenv()

client = None
# client = OpenAI(base_url="https://gpt-oss-120b-svarambally.nrp-nautilus.io/v1", api_key=os.getenv("OSS_KEY"))


def _get_client():
    global client
    if client is not None:
        return client
    api_key = os.getenv("GPT_KEY")
    if not api_key:
        raise RuntimeError("GPT_KEY is required only when --mol_lm GPT-4 is selected")
    client = OpenAI(api_key=api_key)
    return client

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
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


def get_attachment_points(smiles: str) -> dict:
    """Identifies atoms with available hydrogens for substitution."""
    try:
        mol = mol_from_smiles(smiles)
        pts = []

        for atom in mol.GetAtoms():
            h_count = atom.GetTotalNumHs()
            if h_count > 0:
                pts.append({
                    "atom_index": atom.GetIdx(),
                    "element": atom.GetSymbol(),
                    "substitutable_hydrogens": h_count,
                })
        return {"attachment_points": pts, "success": True}
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

FRAGMENTS = {
    **ALKYL_FRAGMENTS,
    **HALOGEN_FRAGMENTS,
    **HETEROATOM_FRAGMENTS,
    **CARBONYL_FRAGMENTS,
    **POLAR_FRAGMENTS
}


def add_functional_group(
    smiles: str,
    target_atom_index: int,
    group: str
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

        rw.AddBond(
            target_atom_index,
            attach_idx_combo,
            Chem.BondType.SINGLE
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


def replace_atom(
    smiles: str,
    atom_index: int,
    new_element: str
) -> dict:
    """
    Replaces an atom with a different element.
    WARNING: This can still create invalid molecules if valence doesn't match.
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


def replace_substructure(
    smiles: str,
    query_smarts: str,
    replacement_smiles: str,
    replace_all: bool = False
) -> dict:
    """
    Replaces a substructure pattern with a new fragment.
    FIX: Added replace_all parameter and better match reporting.
    """
    try:
        mol = mol_from_smiles(smiles)
        query = Chem.MolFromSmarts(query_smarts)
        replacement = mol_from_smiles(replacement_smiles)

        if query is None:
            return {"success": False, "error": "Invalid SMARTS pattern"}

        matches = mol.GetSubstructMatches(query)
        if not matches:
            return {"success": False, "error": "No matching substructure found"}

        replaced = Chem.ReplaceSubstructs(
            mol,
            query,
            replacement,
            replaceAll=replace_all
        )

        if not replaced:
            return {"success": False, "error": "Replacement failed"}

        frags = Chem.GetMolFrags(replaced[0], asMols=True)
        if len(frags) > 1:
            return {
                "success": False, 
                "error": "Replacement resulted in disconnected fragments. The query may not include proper attachment points."
            }
        
        Chem.SanitizeMol(replaced[0])
        return {
            "success": True,
            "new_smiles": canonical_smiles(replaced[0]),
            "num_matches": len(matches),
            "replaced_all": replace_all,
            "warning": "Only first match replaced" if not replace_all and len(matches) > 1 else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_properties(smiles: str) -> dict:
    """Calculates common molecular descriptors."""
    try:
        mol = mol_from_smiles(smiles)
        return {
            "success": True,
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
    
def remove_substructure(
    smiles: str,
    substructure_smarts: str
) -> dict:
    """
    Removes all instances of a substructure match from the molecule.
    Fails if the removal causes the molecule to break into disconnected fragments.
    """
    mol = Chem.MolFromSmiles(smiles)
    query = Chem.MolFromSmarts(substructure_smarts)

    if not mol:
        return {"success": False, "error": "Invalid base SMILES"}
    if not query:
        return {"success": False, "error": "Invalid SMARTS pattern"}

    matches = mol.GetSubstructMatches(query)
    
    if not matches:
        return {"success": False, "error": "Substructure not found in molecule"}

    indices_to_remove = set()
    for match in matches:
        indices_to_remove.update(match)

    try:
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


# ------------------------------------------------------------
# Tool registry
# ------------------------------------------------------------
TOOLS = {
    "validate_smiles": validate_smiles,
    "get_attachment_points": get_attachment_points,
    "add_atom": add_atom,
    "add_functional_group": add_functional_group,
    "replace_atom": replace_atom,
    "replace_substructure": replace_substructure,
    "remove_substructure": remove_substructure,
    "calculate_properties": calculate_properties,
}


# ------------------------------------------------------------
# Tool schemas (OpenAI Responses API)
# ------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "add_atom",
        "description": "Attach atom",
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
        "name": "add_functional_group",
        "description": "Attach group",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "group": {
                    "type": "string",
                    "enum": [
                        "ethyl","propyl","isopropyl","tert_butyl",
                        "cyclopropyl","cyclobutyl","cyclopentyl","cyclohexyl",
                        "fluoro","chloro","bromo","iodo",
                        "hydroxyl","methoxy","ethoxy",
                        "amine","methylamine","dimethylamine",
                        "thiol","methylthio",
                        "aldehyde","ketone_methyl","carboxylic_acid",
                        "ester_methyl","amide","amide_methyl","urea","carbamate",
                        "hydroxymethyl","aminoethyl","dimethylaminoethyl",
                        "morpholine","piperazine","piperidine",
                    ],
                },
            },
            "required": ["smiles", "target_atom_index", "group"],
        },
    },
    {
        "type": "function",
        "name": "replace_atom",
        "description": "Swap element",
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
        "name": "replace_substructure",
        "description": "SMARTS replace",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "query_smarts": {"type": "string"},
                "replacement_smiles": {"type": "string"},
            },
            "required": ["smiles", "query_smarts", "replacement_smiles"],
        },
    },
    {
        "type": "function",
        "name": "remove_substructure",
        "description": "SMARTS delete",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "substructure_smarts": {"type": "string"},
            },
            "required": ["smiles", "substructure_smarts"],
        },
    },
]


# ------------------------------------------------------------
# Agent state
# ------------------------------------------------------------
@dataclass
class AgentState:
    original_smiles: str
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
    initial_smiles: str,
    prompt: str,
    max_steps: int = 10
) -> AgentState:

    state = AgentState(
        original_smiles=initial_smiles,
        current_smiles=initial_smiles
    )

    system_context = (
        "You are a molecular design agent.\n"
        "You may ONLY modify molecules using tools.\n"
        "Only make one modification at a time."
    )
    user_goal = (
        f"Goal: {prompt}\n"
        f"Initial SMILES: {initial_smiles}"
    )
    

    # Reset messages list for this turn
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_goal},
        {"role": "user", "content": f"Possible attachment points: {str(get_attachment_points(initial_smiles))}"},
        {"role": "user", "content": f"Molecule properties: {str(calculate_properties(state.current_smiles))}"}
    ]
    
    modification_tools = ["add_atom", "replace_atom", "add_functional_group", "remove_substructure", "replace_substructure"]

    should_break = False
    for step in range(max_steps):
        print(f"\nCURRENT MESSAGES: {str(messages)}\n")
        response = _get_client().responses.create(
            model='gpt-4.1-mini',
            input=messages,
            tools=TOOL_SCHEMAS
        )
        
        for msg in response.output:
            print(msg)
            if msg.type == "function_call":
                messages.append(msg)
                tool_name = msg.name
                args = json.loads(msg.arguments)

                if "smiles" in args:
                    args["smiles"] = state.current_smiles

                # Execute
                result = execute_tool(tool_name, args)
                print(result)
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
                
                if tool_name in modification_tools:
                    for i, item in enumerate(messages):
                        if isinstance(item, dict) and "content" in item and "Possible attachment points" in item["content"]:
                            messages.pop(i)
                            break

                    messages.append({
                        "role": "user",
                        "content": (
                            "Output FINAL_ANSWER if you have made sufficient modifications (make at most 3). Ensure that desired properties are maintained.\n"
                            f"Current SMILES: {state.current_smiles}\n"
                            f"Possible attachment points: {str(get_attachment_points(state.current_smiles))}"
                        )
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Molecule properties: {str(calculate_properties(state.current_smiles))}"
                    })
                
            if msg.type == "message":
                content = msg.content[0].text
                if "FINAL_ANSWER" in content:
                    state.final_answer = content
                    should_break = True
                    break
        if should_break:
            break
    return state

class GPT4:
    def __init__(self, oracle):
        self.task2description = {
                'cmet': 'I have two molecules and their docking scores to c-MET. The docking score measures how well a molecule binds to c-MET. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n',
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
                'cmet': 'Please propose a new molecule that binds better to c-MET. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
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
        self.oracle = oracle
        self.current_summary = ""

    def edit(self, mating_tuples, mutation_rate, target):
        task = self.task
        if target == "c-met":
            protein = "c-MET"
        elif target == "brd4":
            protein = "BRD4"
        else:
            raise Exception("No target provided")
        
        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_mol = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        try:
            
            #original prompt
            # task_definition = f"I have a molecule and its docking scores to {protein}. The docking score measures how well a molecule binds to {protein}. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n"

            task_objective="Improve binding affinity to the protein kinase c-MET. Only make a few (at most 3) modifications, then respond with FINAL_ANSWER. Do not let molecular weight exceed 700.\n"

            # mol_tuple = ''
            # for i in range(2):
            #     tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ',' + str(-parent_scores[i]) + ']'
            #     mol_tuple = mol_tuple + tu
            # prompt = task_definition + mol_tuple + task_objective + self.requirements
                    
            # print("Prompt: " + prompt, flush=True)
            # messages = [{"role": "user", "content": prompt}]
            # r = query_LLM(messages)
            for i in range(10):
                try:
                    state = run_agent(parent_mol[0], task_objective, max_steps=20)
                    break
                except Exception as e:
                    print(e)
                    continue
                
            # r = r.replace("assistant\n\n", "")
            # print("Response: "  + r, flush=True)
            # proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            # proposed_smiles = proposed_smiles.replace('"', '')
            proposed_smiles = state.current_smiles
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(f"LLM-GENERATED: {proposed_smiles}", flush=True)
            assert proposed_smiles != None
            
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
            new_child = co.crossover(Chem.MolFromSmiles(parent_mol[0]), Chem.MolFromSmiles(parent_mol[1]))
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            if new_child is not None: 
                smiles = Chem.MolToSmiles(new_child, isomericSmiles=False, canonical=True)
                print(f"NON-LLM GENERATED: {smiles}")
                score = self.oracle(smiles)
                new_child = Chem.MolFromSmiles(smiles)
                
                # messages.append({"role": "assistant", "content": smiles})
                # summary_prompt = "Software shows that the ligand "+smiles+" has a binding affinity of "+str(-score)+" kcal/mol.\n"
                # summary_prompt += f"Based on this docking result, update our current accumulated knowledge about {protein}. Briefly summarize the most important details and information about the binding target that we've learned so far. Remember that a more negative binding affinity is better. Disregard the previous output format, and do NOT generate any new molecules at this time."
                # messages.append({"role": "user", "content": summary_prompt})
                # summary = query_LLM(messages)
                # self.current_summary = summary.replace("assistant\n\n", "")
            return (new_child, score)

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
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None
