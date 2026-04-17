import os
import time
import requests
import yaml
import random
import torch
import numpy as np
import hashlib
import tarfile
import tempfile
import uuid
from pathlib import Path
from dataclasses import dataclass
import urllib.request
from rdkit import Chem
from rdkit.Chem import Draw
try:
    import tdc
    from tdc.generation import MolGen
except Exception:
    tdc = None
    MolGen = None
from main.utils.chem import *
import math

from .boltz import calculate_boltz
from .docking import calculate_docking
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

num_gpus = torch.cuda.device_count()
API_URL = os.environ.get("BOLTZ_GATEWAY_URL", "https://boltz-api.nrp-nautilus.io/predict_affinity")
OFFSPRING_SIZE = int(os.environ.get("NAUTILUS_ORACLE_WORKERS", "70"))
REQUEST_TIMEOUT_SEC = int(os.environ.get("BOLTZ_REQUEST_TIMEOUT_SEC", "1800"))
REQUEST_MAX_RETRIES = int(os.environ.get("BOLTZ_REQUEST_MAX_RETRIES", "6"))
REQUEST_RETRY_BASE_SEC = float(os.environ.get("BOLTZ_REQUEST_RETRY_BASE_SEC", "2"))
USE_LOCAL_GPU = os.environ.get("MOLLEO_USE_LOCAL_GPU", "0").lower() in ("1", "true", "yes")
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
MSA_API_KEY_VALUE = os.environ.get("MSA_API_KEY_VALUE") or os.environ.get("NIM_API_KEY")
MSA_CACHE_PREFIX = os.environ.get("MSA_CACHE_PREFIX", "boltz-msa-cache")
NIM_MSA_DATABASE = os.environ.get("NIM_MSA_DATABASE", "Uniref30_2302")
NIM_MSA_TIMEOUT_SEC = int(os.environ.get("NIM_MSA_TIMEOUT_SEC", "180"))
MOLLEO_BOLTZ_CACHE_DIR = Path(os.environ.get("MOLLEO_BOLTZ_CACHE_DIR", "~/.boltz")).expanduser()
use_nautilus = True

_REQUEST_SESSION = requests.Session()
# Avoid inheriting host-level proxy env unless explicitly requested.
if os.environ.get("BOLTZ_USE_ENV_PROXY", "0").lower() not in ("1", "true", "yes"):
    _REQUEST_SESSION.trust_env = False
_S3_CLIENT = None

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
class ArtifactBundle:
    artifact_uri: str | None
    artifact_sha256: str | None
    record_ids: dict[str, str]
    artifact_key: str | None
    failed_ligands: set[str]


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
    # Used only for cache keys; not for security.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324


def _extract_a3m_sequences(a3m_text: str) -> list[str]:
    seqs: list[str] = []
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
    """Return local MSA CSV path, backed by S3 cache when configured.

    This mirrors the naive path where MSA is computed once per protein and then reused.
    """
    if not USE_MSA_SERVER:
        return None
    if protein_name not in PROTEIN_SEQUENCES:
        return None

    protein_seq = PROTEIN_SEQUENCES[protein_name]
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


def _write_input_yaml(
    protein_name: str,
    ligand_smiles: str,
    out_path: Path,
    msa_path: str | None = None,
) -> None:
    if protein_name not in PROTEIN_SEQUENCES:
        raise RuntimeError(f"unsupported protein: {protein_name}")
    protein_entry = {"protein": {"id": "A", "sequence": PROTEIN_SEQUENCES[protein_name]}}
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
    with open(out_path, "w") as f:
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


def _prepare_artifact_bundle(protein_name: str, ligands: list[str], experiment_id: str) -> ArtifactBundle:
    if process_inputs is None:
        raise RuntimeError("boltz process_inputs unavailable; install boltz in client env")
    if not ARTIFACT_BUCKET:
        raise RuntimeError("missing BOLTZ_ARTIFACT_BUCKET (or S3_BUCKET) for client-prep mode")
    if not ligands:
        raise RuntimeError("cannot prepare artifact for empty ligand list")

    _ensure_mol_assets(MOLLEO_BOLTZ_CACHE_DIR)

    shared_msa_csv: Path | None = _get_or_create_shared_msa_csv(protein_name)

    run_token = f"{int(time.time())}_{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix=f"molleo_art_{run_token}_") as tmp:
        tmp_root = Path(tmp)
        in_dir = tmp_root / "yaml"
        out_dir = tmp_root / "prep"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        record_ids: dict[str, str] = {}
        failed_ligands: set[str] = set()
        yaml_paths: list[Path] = []
        for i, lig in enumerate(ligands):
            if Chem.MolFromSmiles(lig) is None:
                failed_ligands.add(lig)
                continue
            rid = f"{experiment_id}_{run_token}_{i:04d}"
            record_ids[lig] = rid
            yp = in_dir / f"{rid}.yaml"
            _write_input_yaml(
                protein_name,
                lig,
                yp,
                msa_path=str(shared_msa_csv.resolve()) if shared_msa_csv else None,
            )
            yaml_paths.append(yp)

        if not yaml_paths:
            return ArtifactBundle(
                artifact_uri=None,
                artifact_sha256=None,
                record_ids={},
                artifact_key=None,
                failed_ligands=failed_ligands,
            )

        use_msa_server = bool(USE_MSA_SERVER)
        msa_server_url = MSA_SERVER_URL
        msa_server_username = MSA_SERVER_USERNAME
        msa_server_password = MSA_SERVER_PASSWORD
        api_key_header = MSA_API_KEY_HEADER
        api_key_value = MSA_API_KEY_VALUE
        if shared_msa_csv is not None:
            # MSA materialized locally; skip remote MSA in process_inputs.
            use_msa_server = False
            msa_server_url = "local://provided-msa"
            msa_server_username = None
            msa_server_password = None
            api_key_header = None
            api_key_value = None

        process_inputs(
            data=yaml_paths,
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

        records_dir = out_dir / "processed" / "records"
        existing = {p.stem for p in records_dir.glob("*.json")} if records_dir.exists() else set()
        ok_record_ids: dict[str, str] = {}
        for lig, rid in record_ids.items():
            if rid in existing:
                ok_record_ids[lig] = rid
            else:
                failed_ligands.add(lig)

        if not ok_record_ids:
            return ArtifactBundle(
                artifact_uri=None,
                artifact_sha256=None,
                record_ids={},
                artifact_key=None,
                failed_ligands=failed_ligands,
            )

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

    return ArtifactBundle(
        artifact_uri=artifact_uri,
        artifact_sha256=artifact_sha,
        record_ids=ok_record_ids,
        artifact_key=artifact_key,
        failed_ligands=failed_ligands,
    )


class _DummyBatchOracle:
    def __call__(self, smis):
        return [0.0 for _ in smis]


class _DummyDiversity:
    def __call__(self, smis):
        return 0.0


class _PassThroughFilter:
    def __call__(self, smis):
        return smis

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
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)) # increasing order
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

def calculate_boltz_nautilus(
    protein_name,
    ligand_smiles,
    idx,
    experiment_id,
    record_id=None,
    artifact_uri=None,
    artifact_sha256=None,
):
    if CLIENT_PREP_MODE and (not record_id or not artifact_uri or not artifact_sha256):
        raise RuntimeError(
            "client-prep contract violation: missing record_id/artifact_uri/artifact_sha256 "
            f"for ligand={ligand_smiles[:40]}"
        )

    print(f"\n[Worker {str(idx)}] Sending job for {ligand_smiles[:60]}...", flush=True)

    payload = {
        "protein_name": protein_name,
        "ligand": ligand_smiles,
        "experiment_id": experiment_id,
    }
    if record_id:
        payload["record_id"] = record_id
    if artifact_uri:
        payload["artifact_uri"] = artifact_uri
    if artifact_sha256:
        payload["artifact_sha256"] = artifact_sha256
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
                return None
            if response.status_code in (429, 502, 503, 504):
                print(
                    f"[Worker {str(idx)}] transient {response.status_code} "
                    f"(attempt {attempt}/{REQUEST_MAX_RETRIES})",
                    flush=True,
                )
            else:
                print(f"[Worker {str(idx)}] Error {response.status_code}: {response.text[:200]}", flush=True)
                return None
        except Exception as e:
            print(
                f"[Worker {str(idx)}] Connection failed (attempt {attempt}/{REQUEST_MAX_RETRIES}): {str(e)}",
                flush=True,
            )
        if attempt < REQUEST_MAX_RETRIES:
            time.sleep(REQUEST_RETRY_BASE_SEC * attempt)

    print(f"[Worker {str(idx)}] Exhausted retries", flush=True)
    return None


def gpu_worker(worker_id, task_q, result_q, evaluator, boltz_cache, experiment_id="default"):
    while True:
        try:
            idx, x, val, record_id, artifact_uri, artifact_sha256 = task_q.get(timeout=3)
            if x is None:
                result_q.put((idx, x, -100))
                print("Buffer is full")
            elif val is not None:
                result_q.put((idx, x, val))
                print("Already in buffer")
            else:
                if boltz_cache and x in boltz_cache:
                    y = -float(boltz_cache[x])
                else:
                    if USE_LOCAL_GPU and num_gpus > 0 and worker_id < num_gpus:
                        y = -float(calculate_boltz(evaluator, x, worker_id))
                    else:
                        result = calculate_boltz_nautilus(
                            evaluator,
                            x,
                            worker_id,
                            experiment_id,
                            record_id=record_id,
                            artifact_uri=artifact_uri,
                            artifact_sha256=artifact_sha256,
                        )
                        y = -float(result) if result is not None else 0
                result_q.put((idx, x, y))
                print(f"\nWorker {worker_id} produced result: {str((x, y))}", flush=True)
        except Empty:
            break
        

class Oracle:
    def __init__(self, args=None, seed=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        self.seed = seed
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log

        self.mol_buffer = mol_buffer
        if tdc is not None:
            self.sa_scorer = tdc.Oracle(name='SA')
            self.diversity_evaluator = tdc.Evaluator(name='Diversity')
        else:
            self.sa_scorer = _DummyBatchOracle()
            self.diversity_evaluator = _DummyDiversity()
        self.last_log = 0

        self.boltz_cache = None
        self.oracle_name=None

    def parallel_oracle(self, inputs):
        print(inputs)
        print("Number of local GPUs available: " + str(num_gpus))

        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()

        # Determine which ligands are truly new (need gateway scoring).
        new_inputs = []
        overflow_indices = set()
        num_new = 0
        for i, x in enumerate(inputs):
            if len(self.mol_buffer) + num_new >= self.max_oracle_calls:
                overflow_indices.add(i)
                continue
            if x not in self.mol_buffer:
                new_inputs.append((i, x))
                num_new += 1

        artifact_bundle = None
        if CLIENT_PREP_MODE and use_nautilus and new_inputs:
            unique_ligs = list(dict.fromkeys(x for _, x in new_inputs))
            exp_id = f"{self.evaluator}_seed{self.seed}" if self.seed is not None else self.evaluator
            t0 = time.time()
            artifact_bundle = _prepare_artifact_bundle(self.evaluator, unique_ligs, exp_id)
            prep_ok = len(artifact_bundle.record_ids)
            prep_fail = len(artifact_bundle.failed_ligands)
            print(
                f"[client-prep] total={len(unique_ligs)} ok={prep_ok} fail={prep_fail} "
                f"artifact_key={artifact_bundle.artifact_key or 'none'} "
                f"sha={(artifact_bundle.artifact_sha256[:12] if artifact_bundle.artifact_sha256 else 'none')} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

        # enqueue tasks
        for i, x in enumerate(inputs):
            if i in overflow_indices:
                task_q.put((i, None, None, None, None, None))
                continue
            if x in self.mol_buffer:
                task_q.put((i, x, self.mol_buffer[x][0], None, None, None))
            else:
                rid = None
                auri = None
                asha = None
                if artifact_bundle is not None:
                    if x in artifact_bundle.failed_ligands:
                        task_q.put((i, x, 0, None, None, None))
                        continue
                    rid = artifact_bundle.record_ids.get(x)
                    if rid:
                        auri = artifact_bundle.artifact_uri
                        asha = artifact_bundle.artifact_sha256
                if CLIENT_PREP_MODE and use_nautilus:
                    if not rid:
                        raise RuntimeError(f"client-prep missing record_id for ligand={x[:40]}")
                    if not auri or not asha:
                        raise RuntimeError(
                            "client-prep missing artifact_uri/artifact_sha256 "
                            f"(artifact_key={artifact_bundle.artifact_key if artifact_bundle else 'none'})"
                        )
                task_q.put((i, x, None, rid, auri, asha))

        # Queue-backed mode: spawn fixed gateway workers (plus optional local GPU workers).
        procs = []
        num_local_workers = num_gpus if USE_LOCAL_GPU else 0
        num_nautilus_workers = OFFSPRING_SIZE if use_nautilus else 0
        num_workers = num_local_workers + num_nautilus_workers
        if num_workers <= 0:
            num_workers = max(len(inputs), 1)
        print(
            f"{str(num_workers)} workers available "
            f"({str(num_local_workers)} local, {str(num_nautilus_workers)} Nautilus gateway)",
            flush=True,
        )
        exp_id = f"{self.evaluator}_seed{self.seed}" if self.seed is not None else self.evaluator
        for worker_id in range(num_workers):
            p = ctx.Process(
                target=gpu_worker,
                args=(worker_id, task_q, result_q, self.evaluator, self.boltz_cache, exp_id),
            )
            p.start()
            procs.append(p)
            time.sleep(0.1)

        # collect results
        results = [None] * len(inputs)
        buffer_length = len(self.mol_buffer)
        num_added = 0
        for i in range(len(inputs)):
            idx, x, y = result_q.get()
            results[idx] = y
            if x not in self.mol_buffer and y!=-100: 
                self.mol_buffer[x] = [y, buffer_length+num_added+1]
                num_added += 1
        for p in procs:
            p.join()
        print("RESULTS: " + str(results))
        print(self.mol_buffer)
        return results


    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results/' + suffix + '.yaml')

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)


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

    def get_docking_data(self, smiles, protein):
            try:
                response = requests.get(f"https://west.ucsd.edu/llm_project/?endpoint=run_docking&smiles={smiles}&target={protein}")
                if response.status_code == 200:
                    data = response.json()
                    if 'error' in data:
                        return 0
                    return data["binding_affinity"]
                else:
                    time.sleep(60)
                    return 0
            except Exception as e:
                print(e)
                time.sleep(60)
                return 0

    def score_smi(self, smi, device):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi in self.mol_buffer:
            print("Already in buffer", flush=True)
            pass
        else:
            print(smi, flush=True)
            fitness = -float(calculate_boltz(self.evaluator, smi, device))
            # fitness = -float(calculate_docking(self.evaluator, smi))
            if fitness < 0.0: fitness = 0.0
            print(fitness, flush=True)
            #print(fitness, type(fitness))
            if math.isnan(fitness):
                fitness = 0
            self.mol_buffer[smi] = [fitness, len(self.mol_buffer)+1]
        return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            print(len(smiles_lst), flush=True)
            
            score_list = self.parallel_oracle(smiles_lst)
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
                run_name = self.args.run_name + "_" + str(self.seed)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None, seed=None):
        self.model_name = args.mol_lm
        self.args = args
        self.seed = seed
        print(self.args.run_name, flush=True)
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args, seed=self.seed)
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            if MolGen is None:
                raise RuntimeError(
                    "tdc is unavailable and --smi_file was not provided; "
                    "please pass --smi_file to run without tdc."
                )
            data = MolGen(name='ZINC')
            print(data)
            self.all_smiles = data.get_data()['smiles'].tolist()
            # self.all_smiles = []
            # with open(f"data/{args.oracles[0]}.txt", "r") as file:
            #     for line in file:
            #         ligand = line[:-1]
            #         self.all_smiles.append(ligand)


        if tdc is not None:
            self.sa_scorer = tdc.Oracle(name='SA')
            self.diversity_evaluator = tdc.Evaluator(name='Diversity')
            self.filter = tdc.chem_utils.oracle.filter.MolFilter(
                filters=['PAINS', 'SureChEMBL', 'Glaxo'],
                property_filters_flag=False,
            )
        else:
            self.sa_scorer = _DummyBatchOracle()
            self.diversity_evaluator = _DummyDiversity()
            self.filter = _PassThroughFilter()

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
    def load_smiles_from_file(self, file_name):
        smiles = []
        with open(file_name) as f:
            for line in f:
                s = line.strip()
                if s:
                    smiles.append(s)
        return smiles

    def sanitize(self, smiles_list, score_list=None):
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
            output_file_path = os.path.join(self.args.output_dir, 'results/' + suffix + '.yaml')

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores),
                np.mean(scores[:10]),
                np.max(scores),
                self.diversity_evaluator(smis),
                np.mean(self.sa_scorer(smis)),
                float(len(smis_pass) / 100),
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish

    def _optimize(self, oracle, config):
        raise NotImplementedError



    def optimize(self, oracle, config, project="test"):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        run_name = self.args.run_name + "_" + str(self.seed)
        self.oracle.task_label = run_name
        if self.seed <= 2:
            init_cache_dir = os.environ.get("MOLLEO_INIT_CACHE_DIR", "/home/ubuntu/MOLLEO/init_caches")
            cache_file = os.path.join(init_cache_dir, f"{oracle}_{str(self.seed)}.yaml")
            if os.path.exists(cache_file):
                with open(cache_file, "r") as file:
                    self.oracle.boltz_cache = yaml.safe_load(file)
                    print(self.oracle.boltz_cache)
            else:
                print(f"[init-cache] missing {cache_file}, continue without cache", flush=True)

        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(run_name)
        self.reset()
