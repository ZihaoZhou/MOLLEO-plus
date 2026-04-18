from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
import torch
import yaml

rdBase.DisableLog('rdApp.error')

import main.molleo.crossover as co, main.molleo.mutate as mu
from main.optimizer import BaseOptimizer

from .utils import get_fp_scores
from .network import create_and_train_network, obtain_model_pred
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor


MINIMUM = 1e-10

def make_mating_pool(population_smiles: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    all_tuples = list(zip(population_scores, population_smiles))
    population_scores = [s + MINIMUM for s in population_scores]
    print(all_tuples, flush=True)
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    print(population_probs, flush=True)
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    print(mating_tuples)
    return mating_tuples


def reproduce(mating_tuples, mutation_rate, mol_lm=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent = []
    parent.append(random.choice(mating_tuples))
    parent.append(random.choice(mating_tuples))

    parent_mol = [t[1] for t in parent]
    new_child = co.crossover(parent_mol[0], parent_mol[1])
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child, new_child_mutation

def get_best_mol(population_scores, population_mol):
    top_mol = population_mol[np.argmax(population_scores)]
    top_smi = Chem.MolToSmiles(top_mol, canonical=True)
    return top_smi

class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None, seed=None):
        super().__init__(args)
        self.model_name = "molleo"

        self.mol_lm = None
        if args.mol_lm == "GPT-4":
            from main.molleo.GPT4 import GPT4
            self.mol_lm = GPT4(self.oracle)
        elif args.mol_lm == "GPT-oss":
            from main.molleo.GPToss import GPToss
            self.mol_lm = GPToss(self.oracle)
        elif args.mol_lm == "custom":
            from main.molleo.custom_llm import Custom_LLM
            # model_path = "/home/ubuntu/LLaMA-Factory/output/llama3_8b_sft_brd4"
            model_path = "meta-llama/Llama-3.1-8B-Instruct"
            print("Model: " + model_path)
            self.mol_lm = Custom_LLM(model_path, self.oracle)
        elif args.mol_lm == "BioT5":
            from main.molleo.biot5 import BioT5
            self.mol_lm = BioT5()
        self.args = args
        self.seed = seed
        lm_name = "baseline"
        if args.mol_lm != None:
            lm_name = args.mol_lm
            self.mol_lm.task = self.args.oracles

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        if self.args.checkpoint_file is not None:
            data = yaml.safe_load(self.args.checkpoint_file)
            population_smiles = sorted(data, key=lambda k: data[k][0], reverse=True)[:config["population_size"]]
            print("Starting from checkpoint. Current population: " + str(population_smiles))
            population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
            population_scores = [data[x][0] for x in population_smiles]
            self.oracle.mol_buffer = data
        else:
            if self.smi_file is not None:
                # Exploitation run
                starting_population = self.all_smiles[:config["population_size"]]
            else:
                # Exploration run
                starting_population = list(np.random.choice(self.all_smiles, config["population_size"]))
                # print(len(self.all_smiles))
                # print(config["population_size"])
                # starting_population = self.all_smiles[:config["population_size"]]
            # select initial population
            
            population_smiles = starting_population
            print("Before sanitation: " + str(len(population_smiles)))
            population_smiles = self.sanitize(population_smiles)
            print("After sanitation: " + str(len(population_smiles)))
            population_scores = self.oracle(population_smiles)

        patience = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # new_population
            mating_tuples = make_mating_pool(population_smiles, population_scores, config["population_size"])

            if self.args.mol_lm == "GPT-4" or self.args.mol_lm == "GPT-oss" or self.args.mol_lm == "custom":
                with ThreadPoolExecutor(max_workers=5) as pool:
                    inputs = [(idx, mating_tuples, config["mutation_rate"], self.oracle.evaluator, self.args.use_tools) for idx in range(config["offspring_size"])]
                    offspring_smiles = list(pool.map(lambda x: self.mol_lm.edit(*x), inputs))
                # offspring_smiles = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.oracle.evaluator) for _ in range(config["offspring_size"])]
                # add new_population
                # offspring_mol, offspring_scores = zip(*offspring_mol_pairs)
                # offspring_mol = list(offspring_mol)
                # offspring_scores = list(offspring_scores)
                
                # population_mol += offspring_mol
                # population_scores += offspring_scores
                # population_mol, population_scores = self.sanitize(population_mol, population_scores)
            elif self.args.mol_lm == "BioT5":
                top_smi = get_best_mol(population_scores, population_mol) 

                offspring_mol = [reproduce(mating_tuples, config["mutation_rate"]) for _ in range(config["offspring_size"])]
                offspring_mol = [item[0] for item in offspring_mol]
                editted_smi = []
                for m in offspring_mol:
                    if m != None:
                        editted_smi.append(Chem.MolToSmiles(m, canonical=True))
                ii = 0
                idxs = np.argsort(population_scores)[::-1]
                print("Bin size: " + str(self.args.bin_size))
                while len(editted_smi) < self.args.bin_size:
                    if ii == len(idxs):
                        print("exiting while loop before filling up bin..........")
                        break
                    m = population_mol[idxs[ii]]
                    editted_mol = self.mol_lm.edit([m])[0]

                    if editted_mol != None:
                        s = Chem.MolToSmiles(editted_mol, canonical=True)
                        if s != None:
                            print("adding editted molecule!!!")
                            editted_smi.append(s)
                    ii += 1
                sim = get_fp_scores(editted_smi, top_smi)
                print("fp_scores_to_top", sim)
                sorted_idx = np.argsort(np.squeeze(sim))[::-1][:config["offspring_size"]]
                print("top 70", sorted_idx)
                editted_smi = np.array(editted_smi)[sorted_idx].tolist()
                offspring_mol = [Chem.MolFromSmiles(s) for s in editted_smi]
                print("len offspring_mol", len(offspring_mol))
                
                population_mol += offspring_mol
                population_mol = self.sanitize(population_mol)
                population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
                
            # population_scores = self.oracle([Chem.MolToSmiles(mol, canonical=True) for mol in population_mol])

            print("Offspring size: " + str(len(offspring_smiles)))
            population_smiles += offspring_smiles
            population_smiles = self.sanitize(population_smiles)
            print("Population size: " + str(len(population_smiles)))
            
            population_scores = self.oracle(population_smiles)
            population_tuples = list(zip(population_scores, population_smiles))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            
            population_smiles = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            
            print("Population Molecules: " + str(population_smiles))
            print("Population Scores: " + str(population_scores))

            ### early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
                
            if self.finish:
                break
