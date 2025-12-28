import json
import os
import threading
from datetime import datetime


STORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "job_store.json"))
_LOCK = threading.Lock()


def _read_store():
    if not os.path.exists(STORE_PATH):
        return {}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_store(data: dict):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_job(job_id: str, dataset_id: str, dataset_path: str, selected_jobs: list, output_dir: str):
    with _LOCK:
        store = _read_store()
        store[job_id] = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "selected_jobs": selected_jobs,
            "output_dir": output_dir,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "results": {}
        }
        _write_store(store)


def update_job(job_id: str, **fields):
    with _LOCK:
        store = _read_store()
        if job_id not in store:
            return
        store[job_id].update(fields)
        _write_store(store)


def get_job(job_id: str):
    with _LOCK:
        store = _read_store()
        return store.get(job_id)


def list_jobs():
    with _LOCK:
        store = _read_store()
        return list(store.values())
