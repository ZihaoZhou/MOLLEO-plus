import base64
import json
import os
import re
import time
import requests
import torch
import yaml
import subprocess

class SingleQuoted(str):
    pass

def single_quoted_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(SingleQuoted, single_quoted_representer)
    
def calculate_boltz(protein_name, ligand):
    if protein_name == "c-met":
        protein_sequence = "XIDLSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVSQFLTEGIIMKDFSXPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKGMKYLASKKFVXRDLAARNCMLDEKFTVKVAXFGLARDMYDKEYYSVXNKTGAKLPVKWMALESLQTQKFTTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWXPKAEMRPSFSELVSRISAIFSTFIG"
    elif protein_name == "brd4":
        protein_sequence = "SHMEQLKCCSGILKEMFAKKHAAYAWPFYKPVDVEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARKLQDVFEMRFAKM"
    else:
        print("Uknown protein!")
        return
    
    msa_exists = os.path.isfile(f"/data/boltz/msa/{protein_name}.csv")
    if msa_exists:
        data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": protein_sequence,
                        "msa": f"/data/boltz/msa/{protein_name}.csv"
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": SingleQuoted(ligand)
                    }
                }
            ],
            "properties": [
                {
                    "affinity": {
                        "binder": "B"
                    }
                }
            ]
        }
    else:
        data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": protein_sequence,
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": SingleQuoted(ligand)
                    }
                }
            ],
            "properties": [
                {
                    "affinity": {
                        "binder": "B"
                    }
                }
            ]
        }
        
    try:
        ligand = re.sub(r'[\\/:\*\?"<>\|]', '_', ligand)
        name = f"{protein_name}_{ligand}"
        output_file = f"/data/boltz/inputs/{name}.yaml" 
        if os.path.isfile(output_file): os.remove(output_file)
        with open(output_file, "w") as outfile:
            yaml.dump(data, outfile, sort_keys=False)
        
        all_gpu_env = os.environ.copy()
        all_gpu_env['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
        gpu = subprocess.run("python3 /home/ubuntu/LLaMA-Factory/find_gpu.py".split(), env=all_gpu_env, capture_output=True, text=True).stdout
        gpu = gpu.replace('\n', '')
        print("Boltz running on GPU " + gpu, flush=True)
        new_env = os.environ.copy()
        new_env['CUDA_VISIBLE_DEVICES'] = gpu
        
        venv = "/data/boltz/.venv/bin/boltz"
        boltz_command = venv + f" predict {output_file} --use_msa_server --output_format pdb --out_dir /data/boltz/results"
        subprocess.run(boltz_command.split(), env=new_env, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        if not msa_exists:
            move_command = f"cp /data/boltz/results/boltz_results_{name}/msa/{name}_0.csv /data/boltz/msa/{protein_name}.csv"
            subprocess.run(move_command.split())
        
        if not os.path.isfile(f"/data/boltz/results/boltz_results_{name}/predictions/{name}/affinity_{name}.json"):
            affinity = 0
        with open(f"/data/boltz/results/boltz_results_{name}/predictions/{name}/affinity_{name}.json", "r") as file:
            result = json.load(file)
            affinity = result["affinity_pred_value"]
            affinity = (6 - float(affinity)) * -1.364
            affinity = round(affinity, 2)
            
        if not os.path.isfile(f"/data/boltz/results/boltz_results_{name}/predictions/{name}/{name}_model_0.pdb"):
            h_bonds = 0
        else:
            move_command = f"cp /data/boltz/results/boltz_results_{name}/predictions/{name}/{name}_model_0.pdb /home/andrew/boltz_pose.pdb"
            subprocess.run(move_command.split())
            chimerax_command = f"chimerax --offscreen --script vis_{protein_name}.cxc --exit"
            result = subprocess.run(chimerax_command.split(), cwd="/home/andrew", capture_output=True, text=True)
            match = re.search(r"(\d+)\s+hydrogen bonds found", result.stdout)
            if match:
                h_bonds = int(match.group(1))
            else:
                h_bonds = 0
            
        return affinity
    except Exception as e:
        print(e)
        return 0
    
    
def get_docking_data(ligand, protein):
    response = requests.get(f"https://west.ucsd.edu/llm_project/?endpoint=run_docking&smiles={ligand}&target={protein}")
    if response.status_code == 200:
        data = response.json()
        if 'images' in data:
            for i, img in enumerate(data['images']):
                open(f'{img}.png', 'wb').write(base64.b64decode(data['images'][img]))
        if 'error' in data:
            return None
        return data
    else:
        time.sleep(60)
        return None
    
# with open("boltz/ligands.txt", "r") as f:
#     smiles_list = [line.strip() for line in f if line.strip()]

# sequence = "XIDLSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVSQFLTEGIIMKDFSXPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKGMKYLASKKFVXRDLAARNCMLDEKFTVKVAXFGLARDMYDKEYYSVXNKTGAKLPVKWMALESLQTQKFTTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWXPKAEMRPSFSELVSRISAIFSTFIG"
# results = {}
# log = open('boltz/log_docking.txt', 'a')
# for idx, smiles in enumerate(smiles_list, start=1):
#     # affinity = calculate_boltz("c-met", sequence, smiles)
#     print(smiles)
#     data = get_docking_data(smiles, "c-met")
#     print(data)
#     affinity = data["binding_affinity"] if data is not None else 0
#     print(smiles + ": " + str(affinity))
#     log.write(smiles + ": " + str(affinity)+"\n")
#     log.flush()
#     results[smiles] = affinity
# print(results)