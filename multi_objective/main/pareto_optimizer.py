import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import QED
import tdc
from tdc.generation import MolGen
from main.utils.chem import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from .boltz import calculate_boltz

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.max_obj = args.max_obj
        self.min_obj = args.min_obj
        self.max_evaluator = None
        self.min_evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.storing_buffer = {}
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        
        self.boltz_cache = {}

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, args):

        self.max_evaluator = []
        self.min_evaluator = []
        for idx in range(len(self.max_obj)):
            if self.max_obj[idx] == "qed":
                self.max_evaluator.append(("qed", QED.qed))
            # eva = tdc.Oracle(name = self.max_obj[idx])
            # self.max_evaluator.append(eva)
        
        for idx in range(len(self.min_obj)):
            if self.min_obj[idx] == "sa":
                self.min_evaluator.append(("sa", sascorer.calculateScore))
            elif self.min_obj[idx] == "c-met":
                self.min_evaluator.append(("c-met", calculate_boltz))
            elif self.min_obj[idx] == "brd4":
                self.min_evaluator.append(("brd4", calculate_boltz))
            # eva = tdc.Oracle(name = self.min_obj[idx])
            # self.min_evaluator.append(eva)

    def evaluate(self, smi):
        print('\n'+smi)
        score = 0
        results = {}
        for eva_tuple in self.max_evaluator:
            eva_name = eva_tuple[0] 
            eva = eva_tuple[1]
            if eva_name == 'qed':
                mol = Chem.MolFromSmiles(smi)
                evaluation = eva(mol)
                score = score + evaluation
            else:
                evaluation = eva(smi)
                score = score + evaluation
            results[eva_name] = evaluation
        for eva_tuple in self.min_evaluator:
            eva_name = eva_tuple[0]
            eva = eva_tuple[1]
            if eva_name == 'sa':
                mol = Chem.MolFromSmiles(smi)
                evaluation = eva(mol)
                score = score + (1 - ((evaluation - 1)/9))
            elif eva_name == "c-met" or eva_name == "brd4":
                scale = 1
                if smi in self.boltz_cache:
                    evaluation = self.boltz_cache[smi]
                else:
                    evaluation = eva(eva_name, smi)
                score = score + -scale * evaluation
                self.boltz_cache[smi] = evaluation
            results[eva_name] = evaluation
        for name in results:
            print(f"{name}: {str(results[name])}")
        print(f"Score: " + str(score))
        return score

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def clean_buffer(self):
        self.storing_buffer = self.storing_buffer | self.mol_buffer
        self.mol_buffer = {}

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        new_buffer = {}
        for smiles in self.mol_buffer:
            new_buffer[smiles] = [self.boltz_cache[smiles], self.mol_buffer[smiles][1]]
        with open(output_file_path, 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m, canonical=True) for m in mols]
                n_calls = len(self.mol_buffer)
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        # try:
        print({
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100, 
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
        })



    def __len__(self):
        return len(self.mol_buffer) 

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            # smi = Chem.MolToSmiles(mol, canonical=True)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluate(smi)), len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
            print("Current buffer: " + str(self.mol_buffer)) 
            self.sort_buffer()
            self.log_intermediate()
            self.last_log = len(self.mol_buffer)
            self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    def select_pareto_front(self, smiles_lst):
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                single_score = []
                for eva_tuple in self.max_evaluator:
                    eva_name = eva_tuple[0] 
                    eva = eva_tuple[1]
                    if eva_name == 'qed':
                        mol = Chem.MolFromSmiles(smi)
                        single_score.append(1 - eva(mol))
                    else:
                        single_score.append(1 - eva(smi))
                for eva_tuple in self.min_evaluator:
                    eva_name = eva_tuple[0]
                    eva = eva_tuple[1]
                    if eva_name == 'sa':
                        mol = Chem.MolFromSmiles(smi)
                        single_score.append((eva(mol) - 1)/9)
                    elif eva_name == "c-met" or eva_name == "brd4":
                        if smi in self.boltz_cache:
                            boltz = self.boltz_cache[smi]
                        else:
                            boltz = eva(eva_name, smi)
                        self.boltz_cache[smi] = boltz
                        single_score.append(boltz)
                score_list.append(single_score)
                if smi not in self.mol_buffer.keys() and len(self.mol_buffer) <= self.max_oracle_calls:
                    self.mol_buffer[smi] = [float(self.evaluate(smi)), len(self.mol_buffer)+1]
                    print("SMILES: " + smi)
                    print("New buffer length: " + str(len(self.mol_buffer)))
                else:
                    print("SMILES: " + smi)
            score_array = np.array(score_list)
            nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
            pareto_front = np.array(smiles_lst)[nds]
            pareto_front_smiles = list(pareto_front)
            print("Pareto front length: " + str(len(pareto_front_smiles)))
            return pareto_front_smiles
        else:
            print('Smiles should be in the list format.')


    @property
    def finish(self):
        print("Length of buffer: " + str(len(self.mol_buffer)))
        print("Max oracle calls: " + str(self.max_oracle_calls))
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            
    def sanitize(self, smiles_list):
        new_smiles_list = []
        smiles_set = set()
        for smiles in smiles_list:
            if smiles is not None:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_smiles_list.append(smiles)
                except ValueError:
                    print('bad smiles')
        return new_smiles_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        

        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        new_buffer = {}
        for smiles in self.oracle.mol_buffer:
            new_buffer[smiles] = [self.oracle.boltz_cache[smiles], self.oracle.mol_buffer[smiles][1]]
        with open(output_file_path, 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)
    

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
            
    def optimize(self, config, seed=0, project="test"):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed 
        self.oracle.task_label = self.args.run_name + "_" + str(seed) if seed!=0 else self.args.run_name
        self._optimize(config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.oracle.task_label)
        self.reset()

