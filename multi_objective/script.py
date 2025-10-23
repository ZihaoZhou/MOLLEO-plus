import os
import sys
import numpy as np
import yaml
from similarity_clustering import cluster
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import DataStructs
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
        
def analyze_results(run_name, limit=False, llm_only=True):
    print("RUN: " + run_name)
    
    llm_ligands = []
    num_errors = 0
    if llm_only:
        with open(f"multi_objective/logs/{run_name}.txt", 'r') as log_file:
            for line in log_file:
                if "1000/1000" in line:
                    break
                if "LLM-GENERATED:" in line:
                    ligand = line.split()[1].strip()
                    llm_ligands.append(ligand)
                if "NUM LLM ERRORS" in line:
                    num_errors += 1
   
    cmet = []
    ligands = {}
    with open(f"multi_objective/results/results_{run_name}.yaml", 'r') as file:
        data = yaml.safe_load(file)
        for ligand, values in data.items():
            if limit and int(values[1]) > 255:
                continue
            if int(values[1]) <= 120:
                cmet.append(ligand)
            else:
                if (llm_only is False or ligand in llm_ligands) and float(values[0])!=0:
                    affin = float(values[0])
                    mol = Chem.MolFromSmiles(ligand)
                    qed = QED.qed(mol)
                    sa = sascorer.calculateScore(mol)
                    ligands[ligand] = [float(values[0]), qed, sa]
                    
    sorted_ligands = sorted(ligands, key=lambda k: ligands[k][0])
    print(cmet)
    print(len(sorted_ligands))
    best_10 = []
    for i in sorted_ligands[:10]:
        best_10.append(ligands[i][0])
    
    c = cluster(sorted_ligands)
    c = sorted(c, key=lambda k: ligands[k][0])
    best_10_cluster = []
    for i in c[:10]:
        best_10_cluster.append(ligands[i][0])
    print("AVG TOP TEN: " + str(np.mean(best_10)))
    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(best_10_cluster)))
    print("BEST: " + str(min(best_10_cluster)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(best_10_cluster)))
    print("BEST 10 LIGANDS (CLUSTERED):")
    qed = []
    sim = []
    sa = []
    num_better_than_threshold = 0
    threshold = -11
    unique = []
    for idx, ligand in enumerate(c):
        if idx < 10:
            mol = Chem.MolFromSmiles(ligand)
            qed_score = QED.qed(mol)
            qed.append(qed_score)
            sa_score = sascorer.calculateScore(mol)
            sa.append(sa_score)
            
            morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
            fingerprint = morgan.GetFingerprint(mol)
            max_sim = 0
            sim_ligand = ""
            for cmet_ligand in cmet:
                cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
                similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                
                if similarity > max_sim:
                    max_sim = similarity
                    sim_ligand = cmet_ligand
            sim.append(max_sim)
            if ligands[ligand][0] < threshold:
                num_better_than_threshold += 1
            if max_sim < 0.5:
                unique.append(ligands[ligand][0])
            # print(ligand)
    print("AVG QED (clustered): " + str(np.mean(qed)))
    print("AVG SA (clustered): " + str(np.mean(sa)))
    print("STDEV QED: " + str(np.std(qed)))
    print("AVG MAX SIM: " + str(np.mean(sim)))
    print("NUMBER OF LLM ERRORS: " + str(num_errors))
    print("NUM BETTER THAN THRESHOLD: " + str(num_better_than_threshold))
    print("UNIQUE GENERATION MEAN: " + str(np.mean(sorted(unique)[:10])))
    print("UNIQUE GENERATION STD: " + str(np.std(sorted(unique)[:10])))
    
    ligands_list = list(ligands.keys())
    score_list = []
    for ligand in ligands_list:
        single_score = []
        single_score.append(ligands[ligand][0])
        mol = Chem.MolFromSmiles(ligand)
        single_score.append(1 - QED.qed(mol))
        single_score.append(sascorer.calculateScore(mol))
        score_list.append(single_score)
    score_array = np.array(score_list)
    nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
    pareto_front = np.array(ligands_list)[nds]
    print()
    print("PARETO FRONT SIZE: " + str(len(pareto_front)))
    print("PARETO FRONT MEAN: " + str(np.mean([ligands[smiles][0] for smiles in pareto_front])))
    print("PARETO FRONT QED: " + str(np.mean([QED.qed(Chem.MolFromSmiles(smiles)) for smiles in pareto_front])))
    print("PARETO FRONT SA: " + str(np.mean([sascorer.calculateScore(Chem.MolFromSmiles(smiles)) for smiles in pareto_front])))
    hv = HV(ref_point=np.array([0.0, 1.0, 10.0]))
    vals = []
    for ligand in pareto_front:
        single_score = []
        single_score.append(ligands[ligand][0])
        mol = Chem.MolFromSmiles(ligand)
        single_score.append(1 - QED.qed(mol))
        single_score.append(sascorer.calculateScore(mol))
        vals.append(single_score)
    hv = hv(np.array(vals))
    print("HYPERVOLUME: " + str(hv))
    
    # affins = [ligands[ligand] for ligand in c]
    # r2 = r2_score(affins, sim)
    # plt.scatter(affins, sim)
    # plt.xlabel("Affinity")
    # plt.ylabel("Similarity")
    # plt.title("BindingDB")
    # plt.xlim(-13, 0)   # x-axis range
    # plt.ylim(0, 1)
    # plt.text(1, 9, f"$R^2 = {r2:.3f}$", fontsize=12, color="blue")  # add R² to plot
    # plt.savefig("/home/ubuntu/MOLLEO/scatter_plot.png")
    # return sorted(unique)[:10]
    return best_10_cluster
# get_all_ligands()
values1 = analyze_results("GPT-4_c-met_base", limit=False, llm_only=True)
# values2 = analyze_results("GPT-4_brd4_boltz", limit=False, llm_only=False)
# _, p = ttest_ind(values1, values2, alternative="less", equal_var=False)
# print(p)
# create_yaml("GPT-4_c-met_zinc")
# set_similarity()

# runs = [filename.replace(".yaml", "") for filename in os.listdir("multi_objective/results") if "custom_c-met" in filename]
# runs.append("GPT-4_c-met_zinc")
# runs.append("GPT-4_c-met_summary")
# runs.append("GPT-4_c-met_base")
# runs.append("GPT-4_brd4_bindingdb")
# print(runs)
# results = {}
# llm_only_res = []
# not_llm_only_res = []
# for run in runs:
#     llm_only = np.mean(analyze_results(run, bindingdb=True, llm_only=True))
#     not_llm_only = np.mean(analyze_results(run, bindingdb=True, llm_only=False))
#     results[run] = llm_only
#     llm_only_res.append(llm_only)
#     not_llm_only_res.append(not_llm_only)
# sorted_results = sorted(results, key=results.get)
# for result in sorted_results:
#     print(f"{result}: {str(results[result])}")
# print("\n\n")
# print("LLM ONLY: ")
# for i in llm_only_res:
#     print(i)
# print("NOT LLM ONLY: ")
# for i in not_llm_only_res:
#     print(i)
# print(np.corrcoef(llm_only_res, not_llm_only_res))

# mol = Chem.MolFromSmiles("Cc1ccc(-c2cc(N)c(=O)n([C@H](C)C(=O)NC3CCCCC3)n2)o1", sanitize=True)
# for i in range(1000):
#     smiles = Chem.MolToSmiles(mol, canonical=True)
#     print(smiles)