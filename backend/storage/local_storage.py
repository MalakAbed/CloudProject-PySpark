import os
import uuid
import shutil


UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))


def ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_file(file_obj, original_filename: str) -> dict:
    """
    Save uploaded file locally and return info.
    """
    ensure_dirs()

    dataset_id = str(uuid.uuid4())
    safe_name = original_filename.replace("/", "_").replace("\\", "_")
    filename = f"{dataset_id}__{safe_name}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file_obj, f)

    return {"dataset_id": dataset_id, "filename": filename, "path": path}
