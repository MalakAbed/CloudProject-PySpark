import os
import subprocess
from datetime import datetime
from typing import Dict, List

from backend.db.job_store import update_job


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")


SCRIPTS = {
    "stats": os.path.join(PROJECT_ROOT, "spark_stats_basic.py"),
    "ml1_regression": os.path.join(PROJECT_ROOT, "spark_ML1_regression.py"),
    "ml2_classification": os.path.join(PROJECT_ROOT, "spark_ML2_classification.py"),
    "ml3_kmeans": os.path.join(PROJECT_ROOT, "spark_ML3_kmeans.py"),
    "ml4_outliers": os.path.join(PROJECT_ROOT, "spark_ML4_outliers.py"),
}


def _run_cmd(cmd: List[str]) -> str:
    """
    Runs a command and returns combined stdout+stderr.
    """
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(out.strip())
    return out.strip()


def run_job(job_id: str, dataset_path: str, selected_jobs: List[str]) -> Dict:
    os.makedirs(OUTPUTS_ROOT, exist_ok=True)
    job_out_dir = os.path.join(OUTPUTS_ROOT, job_id)
    os.makedirs(job_out_dir, exist_ok=True)

    update_job(job_id, status="running", started_at=datetime.utcnow().isoformat() + "Z", output_dir=job_out_dir)

    results = {}

    try:
        # 1) Stats
        if "stats" in selected_jobs:
            out_dir = os.path.join(job_out_dir, "stats")
            cmd = ["python3", SCRIPTS["stats"], "--input", dataset_path, "--output", out_dir]
            logs = _run_cmd(cmd)
            results["stats"] = {"output_dir": out_dir, "log_tail": logs[-1500:]}

        # 2) ML jobs
        if "ml1_regression" in selected_jobs:
            out_dir = os.path.join(job_out_dir, "ML1_regression")
            cmd = ["python3", SCRIPTS["ml1_regression"], "--input", dataset_path, "--output", out_dir]
            logs = _run_cmd(cmd)
            results["ml1_regression"] = {"output_dir": out_dir, "log_tail": logs[-1500:]}

        if "ml2_classification" in selected_jobs:
            out_dir = os.path.join(job_out_dir, "ML2_classification")
            cmd = ["python3", SCRIPTS["ml2_classification"], "--input", dataset_path, "--output", out_dir]
            logs = _run_cmd(cmd)
            results["ml2_classification"] = {"output_dir": out_dir, "log_tail": logs[-1500:]}

        if "ml3_kmeans" in selected_jobs:
            out_dir = os.path.join(job_out_dir, "ML3_kmeans")
            cmd = ["python3", SCRIPTS["ml3_kmeans"], "--input", dataset_path, "--output", out_dir, "--k", "5"]
            logs = _run_cmd(cmd)
            results["ml3_kmeans"] = {"output_dir": out_dir, "log_tail": logs[-1500:]}

        if "ml4_outliers" in selected_jobs:
            out_dir = os.path.join(job_out_dir, "ML4_outliers")
            cmd = ["python3", SCRIPTS["ml4_outliers"], "--input", dataset_path, "--output", out_dir, "--z", "3"]
            logs = _run_cmd(cmd)
            results["ml4_outliers"] = {"output_dir": out_dir, "log_tail": logs[-1500:]}

        update_job(
            job_id,
            status="done",
            finished_at=datetime.utcnow().isoformat() + "Z",
            results=results
        )
        return results

    except Exception as e:
        update_job(
            job_id,
            status="failed",
            finished_at=datetime.utcnow().isoformat() + "Z",
            error=str(e),
            results=results
        )
        raise
