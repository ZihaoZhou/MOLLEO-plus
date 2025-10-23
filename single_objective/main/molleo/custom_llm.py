import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
from rdkit import Chem
import main.molleo.crossover as co, main.molleo.mutate as mu
import random
MINIMUM = 1e-10
import torch
from vllm import LLM, SamplingParams


def query_LLM(model, tokenizer, device, messages: list):
    message = {"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}
    messages.insert(0, message)
    # VLLM 
    # inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
    # sampling_params = SamplingParams(
    #     temperature=0.95,
    #     top_p=0.75,
    #     top_k=50,
    #     max_tokens=1024
    # )
    # outputs = model.generate([inputs], sampling_params)
    # decoded_output = outputs[0].outputs[0].text
    
    # HuggingFace
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    generation_params = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.95,
        "top_p": 0.75,
        "top_k": 50,
        "pad_token_id": tokenizer.eos_token_id
    }

    outputs = model.generate(**inputs, **generation_params)

    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return decoded_output

class Custom_LLM:
    def __init__(self, model_path, oracle):
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
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """
        self.task=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # VLLM
        # self.model = LLM(model=model_path, gpu_memory_utilization=0.7, max_model_len=4096)
        
        # Huggingface
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error_count = 0
        self.current_summary = ""
        self.oracle = oracle
        
        self.initial_context = """During recent years, variety of chemically diverse c-Met kinase inhibitors have been developed by various research groups using rational drug design approach due to availability of crystal structure of catalytic core of the c-Met kinase. In the present review, diverse structural types of c-Met kinase inhibitors have been reported, and an overview of various interactions with amino acid residues of cMet kinase active site have also been included as indicated by the molecular modeling studies and/or X-ray crystallographic studies. Several class I c-Met kinase inhibitors possessing C-6 substituted bicyclic aromatic scaffolds like triazolotriazine, triazolo[4,5-b]pyrazine, triazolo[4,3-b]pyridazine, triazolopyridinone, [1,2,4]triazolo [4,3-a]pyridine, 1H-pyrazolo[3,4-b]pyridine, indazole, purine, pyrazolo[4,3-b]pyridine, imidazo[1,2-a]pyridine, imidazo[4,5-b]pyrazine, etc. have been synthesized. This bicyclic aromatic scaffold generally has an aromatic hinge binding element with a linker like methylene, oxymethylene, aminomethylene, difluoromethylene, sulfonyl, sulfur, sulfoxide or cyclopropyl. Binding mode analysis of class I c-Met kinase inhibitors reveals the importance of nitrogen containing bicyclic aromatic ring with substitution at C-6 in making p-p stacking interactions with Tyr1230 and a hydrogen bond with Asp1222. The linker is considered to be important between the two aromatic scaffolds to wrap around Met1211. The aromatic hinge binder interacts via hydrogen bonding with Met1160 and/or Pro1158. These class I inhibitors have good degree of selectivity towards c-Met kinase but weak inhibitor of c-Met Tyr1230 Fig. 49. Secoiridoid as a c-Met kinase inhibitor. 1132 P.K. Parikh, M.D. Ghate / European Journal of Medicinal Chemistry 143 (2018) 1103e1138 mutants. Based on binding mode analysis, it can also be concluded that to maintain the desired “U” shaped skeleton of the class I inhibitors, selection of a bicyclic nitrogen containing scaffold, linker and aromatic residue is the key to the development of potent and selective class I c-Met kinase inhibitors. Most class II c-Met kinase inhibitors reported so far generally possess a linker “5 atoms regulation” between two aromatic moieties. Majority structural changes are reportedly carried out in the hinge binding aromatic residue and in the linker. Several phenoxy substituted heterocyclic scaffolds including pyridine, pyrimidine, quinoline, quinazoline, imidazo[1,2-b]pyridazine, imidazo[1,2-a]pyridine, furo[2,3-d]pyrimidine, thieno[2,3-d]pyrimidine, etc. have been identified as hinge binding residues for class II inhibitors, which is responsible for key hydrogen bond interaction with Met1160. Class II inhibitors adopt an inactive DFG-out conformation and the linker usually makes hydrogen bonding with the activation loop and C-helix in the N-terminus domain with Asp1222. Another aromatic residue extends into the deep hydrophobic pocket. Designing of inhibitor that optimally adopt an inactive DFG-out conformation and interact with specific residue is the principal method to develop class II inhibitors. Among various class II inhibitors, quinoline based inhibitors had a good degree of success. The current landscape of the development of c-Met kinase inhibitors is outlined in this review, with a focus on rational drug design and structural optimization to pursue selectivity and favorable pharmacokinetic properties. Various novel chemical entities discussed in the present review may provide an opportunity to scientists and researchers of medicinal chemistry discipline to design and develop successful small molecule c-Met kinase inhibitors as anti-cancer agents in future."""

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
            
            #summary
            # if self.current_summary:
            #     # initial context
            #     task_definition = f'We will collaborate on generating a ligand that can bind to kinase {protein} with high binding affinity. Provided below is a summary of our own current accumulated knowledge about the target based on our past trials:\n\n'
            #     prompt = task_definition + self.current_summary + "\n" 
                
                #no initial context
                # task_definition = f'We will collaborate on generating a ligand that can bind to the kinase c-MET with high binding affinity. Provided below is our current accumulated knowledge about the target:\n\n'
                # prompt = task_definition + self.current_summary + "\n" 
            # else:
            #     task_definition = f'We will collaborate on generating a ligand that can bind to kinase {protein} with high binding affinity. Provided below is a description of our current knowledge of strong binders to the target:\n\n'
            #     prompt = task_definition + self.initial_context +"\n"
            # mol_tuple = f'\nAlso provided are two existing molecules and their binding affinities to {protein}:\n'
            # for i in range(2):
            #     tu = '[' + Chem.MolToSmiles(parent_mol[i], canonical=True) + ',' + str(-parent_scores[i]) + ']\n'
            #     mol_tuple = mol_tuple + tu
            # prompt += mol_tuple
            # task_objective = 'First describe what you have learned from the above information. Then propose a new molecule that binds better to '+protein+'. You can either make crossovers and mutations based on the two given molecules or just propose a new molecule based on your knowledge of the protein target. Ensure that your generation is unique from the two input molecules. Follow this exact format for your final answer, character by character: \\box{MOLECULE}, where MOLECULE is your proposed ligand in SMILES format.'
            # prompt += task_objective
            
            # no summary
            # task_definition = f"We will collaborate on generating a ligand for {protein} with high binding affinity. I will give you the output from docking software after each of your attempts. Assume that there are always more improvements to be made to the current ligand.\n"
            # prompt = task_definition + "Provided are two existing molecules and their binding affinities to c-MET:\n"
            # for i in range(2):
            #     tu = '[' + Chem.MolToSmiles(parent_mol[i], canonical=True) + ',' + str(-parent_scores[i]) + ']\n'
            #     mol_tuple = mol_tuple + tu
            # prompt += mol_tuple
            # prompt += "First describe what you have learned from the above information. Then propose a new molecule that binds better to "+protein+". You can either make crossovers and mutations based on the two given molecules or just propose a new molecule based on your knowledge of the protein target. Ensure that your generation is unique from the two input molecules. Follow this exact format for your final answer, character by character: \box{MOLECULE}, where MOLECULE is your proposed ligand in SMILES format."
            
            # original prompt
            task_definition = f"I have two molecules and their docking scores to {protein}. The docking score measures how well a molecule binds to {protein}. A lower docking score generally indicates a stronger or more favorable binding affinity.\n\n"
            task_objective = f'Please propose a new molecule that binds better to {protein}. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n'
            mol_tuple = ''
            for i in range(2):
                tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ',' + str(-parent_scores[i]) + ']'
                mol_tuple = mol_tuple + tu
            prompt = task_definition + mol_tuple + task_objective + self.requirements
            
            print("Prompt: " + prompt, flush=True)
            messages = [{"role": "user", "content": prompt}]
            r = query_LLM(self.model, self.tokenizer, self.device, messages)
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
            # summary_prompt += "Based on this docking result, update our current accumulated knowledge about the protein target. Briefly summarize the most important details and information about the binding target that we've learned so far. Describe what has been effective and what has not. Do NOT generate any new molecules at this time."
            # messages.append({"role": "user", "content": summary_prompt})
            # summary = query_LLM(self.model, self.tokenizer, self.device, messages)
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
                # summary_prompt += "Based on this docking result, update our current accumulated knowledge about the protein target. Briefly summarize the most important details and information about the binding target that we've learned so far. Describe what has been effective and what has not. Do NOT generate any new molecules at this time."
                # messages.append({"role": "user", "content": summary_prompt})
                # summary = query_LLM(self.model, self.tokenizer, self.device, messages)
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