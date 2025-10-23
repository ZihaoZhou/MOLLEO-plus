import traceback
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("GPT_KEY"))
import re
from rdkit import Chem
import main.molleo.crossover as co, main.molleo.mutate as mu
import random
MINIMUM = 1e-10



def query_LLM(messages, model="gpt-4.1-mini", temperature=0.0):
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
    message += messages

    params = {
        "model": model, #Please use your own open engine
        "max_completion_tokens": 4096,
        "messages": message
    }

    for retry in range(3):
        try:
            obj = client.chat.completions.create(**params)
            response = obj.choices[0].message.content
            message.append({"role": "assistant", "content": response})
            break
        except Exception as e:
            traceback.print_exc()
            print(f"{type(e).__name__} {e}")


    print("=>")
    return response

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
            # summary
            # if self.current_summary:
            #     task_definition = f'We will collaborate on generating a ligand that can bind to the kinase {protein} with high binding affinity. Provided below is our current accumulated knowledge about the target:\n\n'
            #     summary_prompt = self.current_summary+"\n"
            #     prompt = task_definition + summary_prompt 
            #     mol_tuple = f'\nAlso provided are two existing molecules and their binding affinities to {protein}:\n'
            #     for i in range(2):
            #         tu = '[' + Chem.MolToSmiles(parent_mol[i], canonical=True) + ',' + str(-parent_scores[i]) + ']\n'
            #         mol_tuple = mol_tuple + tu
            #     prompt += mol_tuple
            # else:
            #     task_definition = f'We will collaborate on generating a ligand that can bind to kinase {protein} with high binding affinity. Provided below are two molecules and their binding affinities to {protein}.\n'
            #     mol_tuple = ''
            #     for i in range(2):
            #         tu = '[' + Chem.MolToSmiles(parent_mol[i], canonical=True) + ',' + str(-parent_scores[i]) + ']\n'
            #         mol_tuple = mol_tuple + tu
            #     prompt = task_definition + mol_tuple
            # task_objective = 'First describe what you have learned from the above information. Then propose a new molecule that binds better to '+protein+'. You can either make crossovers and mutations based on the two given molecules or just propose a new molecule based on your knowledge of the protein target. Ensure that your generation is unique from the two input molecules. Follow this exact format for your final answer, character by character: \\box{MOLECULE}, where MOLECULE is your proposed ligand in SMILES format.'
            # prompt += task_objective
            
            # if self.current_summary:
            #     context = f"First analyze this summary of our current knowledge of {protein}: \n\n" + self.current_summary + "\n\n"
            #     task_definition = f"I have two molecules and their docking scores to {protein}. The docking score measures how well a molecule binds to {protein}. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n"
            #     task_objective = f'Please propose a new molecule that binds better to {protein}. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n'
            #     mol_tuple = ''
            #     for i in range(2):
            #         tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ',' + str(-parent_scores[i]) + ']'
            #         mol_tuple = mol_tuple + tu
            #     prompt = context + task_definition + mol_tuple + task_objective + self.requirements
            # else:
            
            #no summary
            task_definition = f"We will collaborate on generating a ligand for {protein} with high binding affinity. I will give you the output from docking software after each of your attempts. Assume that there are always more improvements to be made to the current ligand.\n"
            prompt = task_definition + "Provided are two existing molecules and their binding affinities to c-MET:\n"
            mol_tuple = ''
            for i in range(2):
                tu = '[' + Chem.MolToSmiles(parent_mol[i], canonical=True) + ',' + str(-parent_scores[i]) + ']\n'
                mol_tuple = mol_tuple + tu
            prompt += mol_tuple
            prompt += "First describe what you have learned from the above information. Then propose a new molecule that binds better to "+protein+". You can either make crossovers and mutations based on the two given molecules or just propose a new molecule based on your knowledge of the protein target. Ensure that your generation is unique from the two input molecules. Follow this exact format for your final answer, character by character: \\box{MOLECULE}, where MOLECULE is your proposed ligand in SMILES format."
            
            #original prompt
            # task_definition = f"I have two molecules and their docking scores to {protein}. The docking score measures how well a molecule binds to {protein}. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n"
            # task_objective = f'Please propose a new molecule that binds better to {protein}. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n'
            # mol_tuple = ''
            # for i in range(2):
            #     tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ',' + str(-parent_scores[i]) + ']'
            #     mol_tuple = mol_tuple + tu
            # prompt = task_definition + mol_tuple + task_objective + self.requirements
                    
            print("Prompt: " + prompt, flush=True)
            messages = [{"role": "user", "content": prompt}]
            r = query_LLM(messages)
            r = r.replace("assistant\n\n", "")
            print("Response: "  + r, flush=True)
            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = proposed_smiles.replace('"', '')
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(f"LLM-GENERATED: {proposed_smiles}", flush=True)
            assert proposed_smiles != None
            score = self.oracle(proposed_smiles)
            new_child = Chem.MolFromSmiles(proposed_smiles)
            
            # messages.append({"role": "assistant", "content": r})
            # summary_prompt = "Software shows that the ligand you generated ("+proposed_smiles+") had a binding affinity of "+str(-score)+" kcal/mol.\n"
            # summary_prompt += f"Update our current accumulated knowledge about {protein}. Briefly summarize the most important details and information about the binding target that we've learned so far. Remember that a more negative binding affinity is better. Disregard the previous output format, and do NOT generate any new molecules at this time."
            # messages.append({"role": "user", "content": summary_prompt})
            # summary = query_LLM(messages)
            # self.current_summary = summary.replace("assistant\n\n", "")

            return (new_child, score)
        except Exception as e:
            traceback.print_exc()
            print(f"{type(e).__name__} {e}")
            self.error_count += 1
            print("NUM LLM ERRORS: " + str(self.error_count), flush=True)
            score = 0
            new_child = co.crossover(parent_mol[0], parent_mol[1])
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