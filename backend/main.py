import os
import uuid
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from backend.jobs.run_spark import run_job
from backend.storage.local_storage import save_upload_file
from backend.db.job_store import create_job, get_job


app = FastAPI(title="CloudProject Backend")

# streamlit runs on different port => allow it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"ok": True, "message": "CloudProject backend is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    # Accept parquet/csv (you can add more)
    allowed = (".parquet", ".csv")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail=f"Only {allowed} supported for now")

    info = save_upload_file(file.file, file.filename)
    return info


@app.post("/run")
def run(dataset_id: str, dataset_path: str, selected_jobs: list = Body(...)):
    """
    selected_jobs example:
    ["stats","ml1_regression","ml2_classification"]
    """
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="dataset_path does not exist on server")

    job_id = str(uuid.uuid4())
    output_dir = ""  
    create_job(job_id, dataset_id, dataset_path, selected_jobs, output_dir)

    def _bg():
        try:
            run_job(job_id, dataset_path, selected_jobs)
        except Exception:
            pass

    threading.Thread(target=_bg, daemon=True).start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "status": job["status"], "error": job["error"]}


@app.get("/results/{job_id}")
def results(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs")
def jobs():
    return list_jobs()
