import os
import sys
import time
import hashlib
import tarfile
import tempfile
import uuid
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import types

import requests
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

# pytdc currently imports `rdkit.six`, removed in newer RDKit.
# Provide a minimal shim to keep Oracle/MolGen imports working.
if "rdkit.six" not in sys.modules:
    six_mod = types.ModuleType("rdkit.six")
    six_mod.iteritems = lambda d: d.items()
    sys.modules["rdkit.six"] = six_mod

import tdc
from tdc.generation import MolGen
from main.utils.chem import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from .boltz import calculate_boltz
from collections import defaultdict
from torch import multiprocessing as mp
from queue import Empty

try:
    import boto3
except Exception:
    boto3 = None

try:
    from boltz.main import process_inputs, MOL_URL, CCD_URL
except Exception:
    process_inputs = None
    MOL_URL = None
    CCD_URL = None

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

num_gpus = torch.cuda.device_count()
API_URL = os.environ.get("BOLTZ_GATEWAY_URL", "https://boltz-api.nrp-nautilus.io/predict_affinity")
OFFSPRING_SIZE = int(os.environ.get("NAUTILUS_ORACLE_WORKERS", "20"))
REQUEST_TIMEOUT_SEC = int(os.environ.get("BOLTZ_REQUEST_TIMEOUT_SEC", "1800"))
REQUEST_MAX_RETRIES = int(os.environ.get("BOLTZ_REQUEST_MAX_RETRIES", "6"))
REQUEST_RETRY_BASE_SEC = float(os.environ.get("BOLTZ_REQUEST_RETRY_BASE_SEC", "2"))
CLIENT_PREP_MODE = os.environ.get("BOLTZ_CLIENT_PREP_MODE", "1").lower() in ("1", "true", "yes")
ARTIFACT_BUCKET = os.environ.get("BOLTZ_ARTIFACT_BUCKET") or os.environ.get("S3_BUCKET")
ARTIFACT_PREFIX = os.environ.get("BOLTZ_ARTIFACT_PREFIX", "molleo-fast-artifacts")
ARTIFACT_PRESIGN_EXPIRE_SEC = int(os.environ.get("BOLTZ_ARTIFACT_PRESIGN_EXPIRE_SEC", str(24 * 3600)))
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL", "")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
PREP_THREADS = int(os.environ.get("BOLTZ_PREPROCESS_THREADS", "4"))
MSA_SERVER_URL = os.environ.get("BOLTZ_MSA_SERVER_URL") or os.environ.get(
    "MSA_PROXY_URL",
    "http://boltz-msa-proxy.spatiotemporal-decision-making.svc.cluster.local:8000",
)
MSA_PAIRING_STRATEGY = os.environ.get("BOLTZ_MSA_PAIRING_STRATEGY", "greedy")
MAX_MSA_SEQS = int(os.environ.get("BOLTZ_MAX_MSA_SEQS", "8192"))
USE_MSA_SERVER = os.environ.get("BOLTZ_USE_MSA_SERVER", "1").lower() in ("1", "true", "yes")
MSA_SERVER_USERNAME = os.environ.get("BOLTZ_MSA_USERNAME")
MSA_SERVER_PASSWORD = os.environ.get("BOLTZ_MSA_PASSWORD")
MSA_API_KEY_HEADER = os.environ.get("MSA_API_KEY_HEADER", "X-API-Key")
MSA_API_KEY_VALUE = os.environ.get("MSA_API_KEY_VALUE")
MSA_CACHE_PREFIX = os.environ.get("MSA_CACHE_PREFIX", "boltz-msa-cache")
NIM_MSA_DATABASE = os.environ.get("NIM_MSA_DATABASE", "Uniref30_2302")
NIM_MSA_TIMEOUT_SEC = int(os.environ.get("NIM_MSA_TIMEOUT_SEC", "180"))
MOLLEO_BOLTZ_CACHE_DIR = Path(os.environ.get("MOLLEO_BOLTZ_CACHE_DIR", "~/.boltz")).expanduser()
use_nautilus = True
use_local = False
_REQUEST_SESSION = requests.Session()
if os.environ.get("BOLTZ_USE_ENV_PROXY", "0").lower() not in ("1", "true", "yes"):
    _REQUEST_SESSION.trust_env = False
_S3_CLIENT = None
_ARTIFACT_PAYLOAD_CACHE = {}

PROTEIN_SEQUENCES = {
    "c-met": "HIDLSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVSQFLTEGIIMKDFSXPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKGMKYLASKKFVXRDLAARNCMLDEKFTVKVAXFGLARDMYDKEYYSVXNKTGAKLPVKWMALESLQTQKFTTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWXPKAEMRPSFSELVSRISAIFSTFIG",
    "brd4": "SHMEQLKCCSGILKEMFAKKHAAYAWPFYKPVDVEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARKLQDVFEMRFAKM",
}


class SingleQuoted(str):
    pass


def _sq_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")


yaml.add_representer(SingleQuoted, _sq_representer)


@dataclass
class ArtifactPayload:
    record_id: str
    artifact_uri: str
    artifact_sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324


def _extract_a3m_sequences(a3m_text: str) -> list[str]:
    seqs = []
    for line in a3m_text.splitlines():
        s = line.strip()
        if not s or s.startswith(">"):
            continue
        seqs.append(s)
    return seqs


def _fetch_proxy_a3m(protein_name: str, sequence: str) -> str:
    base = (MSA_SERVER_URL or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("missing BOLTZ_MSA_SERVER_URL for proxy MSA provider")
    url = f"{base}/a3m"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if MSA_API_KEY_VALUE:
        headers[MSA_API_KEY_HEADER] = MSA_API_KEY_VALUE
    payload = {
        "protein_name": protein_name,
        "sequence": sequence,
        "database": NIM_MSA_DATABASE,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=NIM_MSA_TIMEOUT_SEC)
    if resp.status_code != 200:
        raise RuntimeError(f"MSA proxy request failed: status={resp.status_code} body={resp.text[:500]}")
    data = resp.json()
    a3m = (data.get("a3m") or "").strip()
    if not a3m:
        raise RuntimeError("MSA proxy response missing a3m")
    return a3m


def _msa_cache_key(protein_name: str, protein_seq: str) -> str:
    seq_hash = _sha1_text(protein_seq)[:10]
    prefix = str(MSA_CACHE_PREFIX).rstrip("/")
    return f"{prefix}/{protein_name}/proxy/{NIM_MSA_DATABASE}/{seq_hash}.csv"


def _get_or_create_shared_msa_csv(protein_name: str) -> Path | None:
    if not USE_MSA_SERVER:
        return None
    protein_seq = PROTEIN_SEQUENCES.get(protein_name)
    if not protein_seq:
        return None
    key = _msa_cache_key(protein_name, protein_seq)
    local_dir = MOLLEO_BOLTZ_CACHE_DIR / "msa_cache" / protein_name / "proxy" / NIM_MSA_DATABASE
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / Path(key).name
    if local.exists():
        return local

    a3m = _fetch_proxy_a3m(protein_name, protein_seq)
    seqs = _extract_a3m_sequences(a3m)
    if not seqs:
        raise RuntimeError("MSA proxy returned empty alignment")
    seqs = seqs[: max(1, int(MAX_MSA_SEQS))]
    with local.open("w") as f:
        f.write("key,sequence\n")
        for s in seqs:
            f.write(f"-1,{s}\n")
    return local


def _ensure_mol_assets(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = cache_dir / "mols"
    tar_mols = cache_dir / "mols.tar"
    ccd_pkl = cache_dir / "ccd.pkl"
    if MOL_URL is None:
        raise RuntimeError("boltz MOL_URL unavailable; cannot bootstrap mol assets")
    if not mol_dir.exists():
        if not tar_mols.exists():
            urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache_dir)  # noqa: S202
    if not ccd_pkl.exists():
        if CCD_URL is None:
            raise RuntimeError("boltz CCD_URL unavailable; cannot bootstrap ccd.pkl")
        urllib.request.urlretrieve(CCD_URL, str(ccd_pkl))  # noqa: S310


def _write_input_yaml(protein_name: str, ligand_smiles: str, out_path: Path, msa_path: str | None = None) -> None:
    protein_seq = PROTEIN_SEQUENCES.get(protein_name)
    if not protein_seq:
        raise RuntimeError(f"unsupported protein: {protein_name}")
    protein_entry = {"protein": {"id": "A", "sequence": protein_seq}}
    if msa_path:
        protein_entry["protein"]["msa"] = msa_path
    data = {
        "version": 1,
        "sequences": [
            protein_entry,
            {"ligand": {"id": "B", "smiles": SingleQuoted(ligand_smiles)}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    with out_path.open("w") as f:
        yaml.dump(data, f, sort_keys=False)


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    if boto3 is None:
        raise RuntimeError("boto3 is missing; install boto3 for client-prep mode")
    _S3_CLIENT = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT or None,
        region_name=S3_REGION,
    )
    return _S3_CLIENT


def _prepare_single_artifact(protein_name: str, ligand_smiles: str, experiment_id: str) -> ArtifactPayload:
    if process_inputs is None:
        raise RuntimeError("boltz process_inputs unavailable; install boltz in client env")
    if not ARTIFACT_BUCKET:
        raise RuntimeError("missing BOLTZ_ARTIFACT_BUCKET (or S3_BUCKET) for client-prep mode")
    if Chem.MolFromSmiles(ligand_smiles) is None:
        raise RuntimeError("invalid ligand smiles")

    _ensure_mol_assets(MOLLEO_BOLTZ_CACHE_DIR)
    shared_msa_csv = _get_or_create_shared_msa_csv(protein_name)

    run_token = f"{int(time.time())}_{uuid.uuid4().hex[:10]}"
    record_id = f"{experiment_id}_{protein_name}_{_sha1_text(ligand_smiles)[:12]}"
    with tempfile.TemporaryDirectory(prefix=f"molleo_art_{run_token}_") as tmp:
        tmp_root = Path(tmp)
        in_dir = tmp_root / "yaml"
        out_dir = tmp_root / "prep"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = in_dir / f"{record_id}.yaml"
        _write_input_yaml(
            protein_name,
            ligand_smiles,
            yaml_path,
            msa_path=str(shared_msa_csv.resolve()) if shared_msa_csv else None,
        )

        use_msa_server = bool(USE_MSA_SERVER)
        msa_server_url = MSA_SERVER_URL
        msa_server_username = MSA_SERVER_USERNAME
        msa_server_password = MSA_SERVER_PASSWORD
        api_key_header = MSA_API_KEY_HEADER
        api_key_value = MSA_API_KEY_VALUE
        if shared_msa_csv is not None:
            use_msa_server = False
            msa_server_url = "local://provided-msa"
            msa_server_username = None
            msa_server_password = None
            api_key_header = None
            api_key_value = None

        process_inputs(
            data=[yaml_path],
            out_dir=out_dir,
            ccd_path=MOLLEO_BOLTZ_CACHE_DIR / "ccd.pkl",
            mol_dir=MOLLEO_BOLTZ_CACHE_DIR / "mols",
            msa_server_url=msa_server_url,
            msa_pairing_strategy=MSA_PAIRING_STRATEGY,
            max_msa_seqs=MAX_MSA_SEQS,
            use_msa_server=use_msa_server,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            api_key_header=api_key_header,
            api_key_value=api_key_value,
            boltz2=True,
            preprocessing_threads=max(1, PREP_THREADS),
        )

        record_json = out_dir / "processed" / "records" / f"{record_id}.json"
        if not record_json.exists():
            raise RuntimeError(f"missing processed record for {record_id}")

        tar_path = tmp_root / f"{run_token}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(out_dir / "processed", arcname="processed")
        artifact_sha = _sha256_file(tar_path)

        artifact_key = f"{ARTIFACT_PREFIX.rstrip('/')}/{experiment_id}/{run_token}.tar.gz"
        s3 = _get_s3_client()
        s3.upload_file(str(tar_path), ARTIFACT_BUCKET, artifact_key)
        artifact_uri = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": ARTIFACT_BUCKET, "Key": artifact_key},
            ExpiresIn=ARTIFACT_PRESIGN_EXPIRE_SEC,
        )

    return ArtifactPayload(record_id=record_id, artifact_uri=artifact_uri, artifact_sha256=artifact_sha)

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

def tuple_to_score(input_tuple, *args):
    for idx, element in enumerate(input_tuple):
        if args[idx][0] == "c-met" or args[idx][0] == "brd4":
            affin = -(element/13)
        elif args[idx][0] == "qed":
            qed = (1 - element)
        elif args[idx][0] == "sa":
            sa = (1 - element)
        else:
            raise Exception("Invalid evaluator")
        
    bias = {"qed": 1, "sa": 1, "affin": 5}
    weights = {"qed": 1, "sa": 1, "affin": 1}

    qed = qed * bias["qed"] * weights["qed"]
    sa = sa * bias["sa"] * weights["sa"]
    affin = affin * bias["affin"] * weights["affin"]
    
    return affin + qed + sa

def calculate_boltz_nautilus(protein_name, ligand_smiles, idx):
    print(f"\n[Worker {str(idx)}] Sending job for {ligand_smiles[:60]}...", flush=True)

    payload = {
        "protein_name": protein_name,
        "ligand": ligand_smiles,
        "experiment_id": os.environ.get("BOLTZ_EXPERIMENT_ID", "molleo_multi"),
    }
    if CLIENT_PREP_MODE:
        cache_key = (protein_name, ligand_smiles)
        artifact = _ARTIFACT_PAYLOAD_CACHE.get(cache_key)
        if artifact is None:
            artifact = _prepare_single_artifact(
                protein_name=protein_name,
                ligand_smiles=ligand_smiles,
                experiment_id=payload["experiment_id"],
            )
            _ARTIFACT_PAYLOAD_CACHE[cache_key] = artifact
        payload["record_id"] = artifact.record_id
        payload["artifact_uri"] = artifact.artifact_uri
        payload["artifact_sha256"] = artifact.artifact_sha256

    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            before = time.time()
            response = _REQUEST_SESSION.post(
                API_URL,
                json=payload,
                headers={"Connection": "close"},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            after = time.time()
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                print(f"[Worker {str(idx)}] {status} in {after-before:.1f}s", flush=True)
                if status == "success":
                    return result.get("affinity", 0)
                print(f"[Worker {str(idx)}] Fail: {result.get('error', 'unknown')}", flush=True)
                return 0
            if response.status_code in (429, 502, 503, 504):
                print(
                    f"[Worker {str(idx)}] transient {response.status_code} "
                    f"(attempt {attempt}/{REQUEST_MAX_RETRIES})",
                    flush=True,
                )
            else:
                print(f"[Worker {str(idx)}] Error {response.status_code}: {response.text[:200]}", flush=True)
                return 0
        except Exception as e:
            print(
                f"[Worker {str(idx)}] Connection failed (attempt {attempt}/{REQUEST_MAX_RETRIES}): {str(e)}",
                flush=True,
            )
        if attempt < REQUEST_MAX_RETRIES:
            time.sleep(REQUEST_RETRY_BASE_SEC * attempt)
    print(f"[Worker {str(idx)}] Exhausted retries", flush=True)
    return 0

def gpu_worker(gpu_id, task_q, result_q, max_evaluator, min_evaluator, boltz_cache):
    while True:
        try:
            idx, smi, val = task_q.get(timeout=3)
            try:
                if smi is None:
                    zeros_max = [0] * len(max_evaluator)
                    zeros_min = [0] * len(min_evaluator)
                    result_q.put((idx, smi, zeros_max + zeros_min, zeros_max + zeros_min))  
                else:
                    single_score = []
                    boltz_scores = []
                    for eva_tuple in max_evaluator:
                        eva_name = eva_tuple[0] 
                        eva = eva_tuple[1]
                        if eva_name == 'qed':
                            mol = Chem.MolFromSmiles(smi)
                            single_score.append(1 - eva(mol))
                        elif eva_name == "c-met" or eva_name == "brd4":
                            if val is not None or smi in boltz_cache[eva_name]:
                                boltz = boltz_cache[eva_name][smi]
                            else:
                                if gpu_id >= num_gpus or not use_local:
                                    boltz = calculate_boltz_nautilus(eva_name, smi, gpu_id)
                                else:
                                    boltz = calculate_boltz(eva_name, smi, gpu_id)
                            boltz_scores.append(boltz)
                            single_score.append(-boltz)
                        else:
                            single_score.append(1 - eva(smi))
                    for eva_tuple in min_evaluator:
                        eva_name = eva_tuple[0]
                        eva = eva_tuple[1]
                        if eva_name == 'sa':
                            mol = Chem.MolFromSmiles(smi)
                            single_score.append((eva(mol) - 1)/9)
                        elif eva_name == "c-met" or eva_name == "brd4":
                            if val is not None or smi in boltz_cache[eva_name]:
                                boltz = boltz_cache[eva_name][smi]
                            else:
                                if gpu_id >= num_gpus or not use_local:
                                    boltz = calculate_boltz_nautilus(eva_name, smi, gpu_id)
                                else:
                                    boltz = calculate_boltz(eva_name, smi, gpu_id)
                            boltz_scores.append(boltz)
                            single_score.append(boltz)
                    result_q.put((idx, smi, single_score, boltz_scores))
                    if val is None: print(f"GPU {gpu_id} produced result: {str((smi, single_score, boltz_scores))}\n")
            except Exception as e:
                print(e)
                sys.exit()
        except Empty:
            break
    
class Oracle:
    def __init__(self, args=None):
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
        self.mol_buffer = {}
        self.storing_buffer = {}
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        
        self.boltz_cache = defaultdict(dict)
        self.starting_population = []

    @property
    def budget(self):
        return self.max_oracle_calls
    
    def parallel_oracle(self, inputs):
        print(inputs)
        print("Number of GPUs available: " + str(num_gpus))

        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()

        # enqueue tasks
        num_added = 0
        for i, x in enumerate(inputs):
            if len(self.mol_buffer) + num_added >= self.max_oracle_calls:
                task_q.put((i, None, None))
                continue
            if x in self.mol_buffer:
                task_q.put((i, x, self.mol_buffer[x][0]))
            else:
                task_q.put((i, x, None))
                num_added += 1
        
        procs = []
        num_nautilus = OFFSPRING_SIZE if use_nautilus else 0
        num_workers = 0
        num_workers += num_gpus if use_local else 0
        num_workers += num_nautilus if use_nautilus else 0
        if num_workers <= 0:
            num_workers = max(1, min(len(inputs), 8))
        print(f"{str(num_workers)} workers available ({str(num_gpus)} local, {str(num_nautilus)} gateway workers)")
        for gpu_id in range(num_workers):
            p = ctx.Process(target=gpu_worker, args=(gpu_id, task_q, result_q, self.max_evaluator, self.min_evaluator, self.boltz_cache))
            p.start()
            procs.append(p)
            time.sleep(0.1)

        # collect results
        results = [None] * len(inputs)
        buffer_length = len(self.mol_buffer)
        num_added = 0
        for i in range(len(inputs)):
            idx, x, single_score, boltz_scores = result_q.get()
            
            # results array
            results[idx] = single_score
            
            # boltz cache
            boltz_index = 0
            for eva_tuple in self.max_evaluator:
                if eva_tuple[0] == "c-met" or eva_tuple[0] == "brd4":
                    self.boltz_cache[eva_tuple[0]][x] = boltz_scores[boltz_index]
                    boltz_index += 1
            for eva_tuple in self.min_evaluator:
                if eva_tuple[0] == "c-met" or eva_tuple[0] == "brd4":
                    self.boltz_cache[eva_tuple[0]][x] = boltz_scores[boltz_index]
                    boltz_index += 1
            
            # mol buffer
            if x not in self.mol_buffer and any(single_score): 
                self.mol_buffer[x] = [tuple_to_score(single_score, *self.max_evaluator, *self.min_evaluator), buffer_length+num_added+1]
                num_added += 1
                
        for p in procs:
            p.join()
        print("RESULTS: " + str(results))
        print(self.mol_buffer)
        return results
    
    def calculate_similarity(self, smiles):
        morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = morgan.GetFingerprint(mol)
        max_sim = 0
        for starting_ligand in self.starting_population:
            cmet_mol = Chem.MolFromSmiles(starting_ligand)
            cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
            similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
            max_sim = max(max_sim, similarity)
        return max_sim

    def assign_evaluator(self, args):

        self.max_evaluator = []
        self.min_evaluator = []
        for idx in range(len(self.max_obj)):
            if self.max_obj[idx] == "qed":
                self.max_evaluator.append(("qed", QED.qed))
            elif self.max_obj[idx] == "c-met":
                self.max_evaluator.append(("c-met", calculate_boltz))
            elif self.max_obj[idx] == "brd4":
                self.max_evaluator.append(("brd4", calculate_boltz))
            # eva = tdc.Oracle(name = self.max_obj[idx])
            # self.max_evaluator.append(eva)
        
        for idx in range(len(self.min_obj)):
            if self.min_obj[idx] == "sa":
                self.min_evaluator.append(("sa", sascorer.calculateScore))
            elif self.min_obj[idx] == "sim":
                self.min_evaluator.append(("sim", self.calculate_similarity))
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
            elif eva_name == "c-met" or eva_name == "brd4":
                scale = 1
                if smi in self.boltz_cache[eva_name]:
                    evaluation = self.boltz_cache[eva_name][smi]
                else:
                    evaluation = eva(eva_name, smi)
                self.boltz_cache[eva_name][smi] = evaluation
                score = score + scale * evaluation
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
                if smi in self.boltz_cache[eva_name]:
                    evaluation = self.boltz_cache[eva_name][smi]
                else:
                    evaluation = eva(eva_name, smi)
                self.boltz_cache[eva_name][smi] = evaluation
                score = score + -scale * evaluation
            elif eva_name == "sim":
                if smi in self.starting_population:
                    evaluation = 0.5
                    score = score + evaluation
                else:
                    evaluation = eva(smi)
                    score = score + (1 - evaluation)
            results[eva_name] = evaluation
        for name in results:
            print(f"{name}: {str(results[name])}")
        print(f"Score: " + str(score))
        # print(self.boltz_cache)
        return score

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def clean_buffer(self):
        self.storing_buffer = self.storing_buffer | self.mol_buffer
        self.mol_buffer = {}

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'seed_' + suffix + ".yaml")

        self.sort_buffer()
        new_buffer = {}
        main_target = ""
        for target in list(self.boltz_cache.keys()):
            if target in self.min_obj:
                main_target = target
                break
        print("Main target: " + main_target)
        for smiles in self.mol_buffer:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
            new_buffer[smiles] = [self.boltz_cache[main_target][smiles], self.mol_buffer[smiles][1], qed, sa]
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
            score_tuples = self.parallel_oracle(smiles_lst)
            score_list = [tuple_to_score(score_tuple, *self.max_evaluator, *self.min_evaluator) for score_tuple in score_tuples]
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

    def crowding_distance(self, scores, front):
        distances = [0.0] * len(front)
        num_obj = len(scores[0])

        for i in range(num_obj):
            sorted_idx = sorted(range(len(front)), key=lambda x: scores[front[x]][i])
            min_val = scores[front[sorted_idx[0]]][i]
            max_val = scores[front[sorted_idx[-1]]][i]
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            for j in range(1, len(front) - 1):
                prev_val = scores[front[sorted_idx[j - 1]]][i]
                next_val = scores[front[sorted_idx[j + 1]]][i]
                if max_val - min_val == 0:
                    d = 0.0
                else:
                    d = (next_val - prev_val) / (max_val - min_val)
                distances[sorted_idx[j]] += d
        return distances
    

    def select_pareto_front(self, smiles_lst):
        if type(smiles_lst) == list:
            score_list = self.parallel_oracle(smiles_lst)
            score_array = np.array(score_list)
            n = 3
            print(f"Taking top {str(n)} fronts")
            nds = NonDominatedSorting().do(score_array, only_non_dominated_front=False)[:n]
            print(nds)
            pareto_front_smiles = []
            num_appended = 0
            for i in range(len(nds)):            
                if num_appended >= 120:
                    break
                print(len(nds[i]))
                if len(nds[i]) + num_appended <= 120:
                    pareto_front_smiles.append(list(np.array(smiles_lst)[nds[i]]))
                    num_appended += len(nds[i])
                else:
                    # crowding distance
                    try:
                        cd = self.crowding_distance(score_list, nds[i])
                        front = list(np.array(smiles_lst)[nds[i]])
                        sorted_front = sorted(front, key=lambda x: cd[front.index(x)], reverse=True)
                        pareto_front_smiles.append(sorted_front[:(120-num_appended)])
                    except Exception as e:
                        print(e)
                        pareto_front_smiles.append(list(np.array(smiles_lst)[nds[i]])[:(120-num_appended)])
                    num_appended += len(nds[i])
            print("Pareto front length: " + str(len(pareto_front_smiles)))
            print("Non dominated front: " + str(list(np.array(smiles_lst)[nds[0]])))
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
        self.all_smiles = []
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:  
            if args.starting == "zinc":      
                data = MolGen(name = 'ZINC')
                self.all_smiles = data.get_data()['smiles'].tolist()
            else:
                with open(f"data/{args.min_obj[0]}.txt", "r") as file:
                    for line in file:
                        ligand = line[:-1]
                        self.all_smiles.append(ligand)
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        # MolFilter is unused in the current optimizer path and can trigger
        # slow runtime dependency installs inside benchmark pods.
        self.filter = None

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
                    if not mol:
                        continue
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
            output_file_path = os.path.join(self.args.output_dir, 'seed_' + suffix + '.yaml')

        self.sort_buffer()
        new_buffer = {}
        main_target = ""
        for target in list(self.oracle.boltz_cache.keys()):
            if target in self.oracle.min_obj:
                main_target = target
                break
            
        for smiles in self.oracle.mol_buffer:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
            new_buffer[smiles] = [self.oracle.boltz_cache[main_target][smiles], self.oracle.mol_buffer[smiles][1], qed, sa]
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
        self.oracle.task_label = str(seed)

        # Initial boltz values (optional warm cache): probe configured/local legacy paths.
        if self.seed <= 4:
            init_roots = []
            env_root = os.environ.get("MOLLEO_INIT_CACHE_DIR", "").strip()
            if env_root:
                init_roots.append(Path(env_root))
            init_roots.append(Path(__file__).resolve().parents[2] / "init_caches")
            init_roots.append(Path("/home/ubuntu/MOLLEO/init_caches"))

            def _try_load_init_cache(eva_name: str) -> None:
                if eva_name not in ("c-met", "brd4"):
                    return
                for root in init_roots:
                    cache_path = root / f"{eva_name}_{self.seed}.yaml"
                    if cache_path.exists():
                        with cache_path.open("r") as file:
                            self.oracle.boltz_cache[eva_name] = yaml.safe_load(file)
                        print(f"Loaded init cache: {cache_path}")
                        return
                print(f"Init cache missing for {eva_name}_{self.seed}; continuing without warm cache")

            for eva in self.args.min_obj:
                _try_load_init_cache(eva)
            for eva in self.args.max_obj:
                _try_load_init_cache(eva)
                
        self._optimize(config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.oracle.task_label)
        self.reset()
