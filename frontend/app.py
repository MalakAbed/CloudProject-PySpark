import time
import json
import requests
import streamlit as st

BACKEND = "http://127.0.0.1:8000"

st.title("CloudProject (Upload → Spark → Results)")

st.write("Backend:", BACKEND)

st.header("1) Upload dataset")
uploaded = st.file_uploader("Upload .parquet or .csv", type=["parquet", "csv"])

dataset_info = st.session_state.get("dataset_info")

if uploaded is not None:
    if st.button("Upload to backend"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        r = requests.post(f"{BACKEND}/upload", files=files)
        if r.status_code != 200:
            st.error(r.text)
        else:
            dataset_info = r.json()
            st.session_state["dataset_info"] = dataset_info
            st.success("Uploaded ✅")
            st.json(dataset_info)

if dataset_info:
    st.header("2) Choose what to run")
    st.write("Uploaded file path on server:")
    st.code(dataset_info["path"])

    opts = []
    if st.checkbox("Descriptive stats (stats)", value=True):
        opts.append("stats")
    if st.checkbox("ML1 Regression (predict total_amount)", value=True):
        opts.append("ml1_regression")
    if st.checkbox("ML2 Classification (card vs cash)", value=True):
        opts.append("ml2_classification")
    if st.checkbox("ML3 KMeans clustering", value=True):
        opts.append("ml3_kmeans")
    if st.checkbox("ML4 Outlier detection", value=True):
        opts.append("ml4_outliers")

    if st.button("Run Spark jobs"):
        payload = {
            "dataset_id": dataset_info["dataset_id"],
            "dataset_path": dataset_info["path"],
            "selected_jobs": opts
        }
        r = requests.post(f"{BACKEND}/run", params={}, json=None, data=None)
        # FastAPI endpoint expects query/body? we used normal params in signature.
        # easiest: send as query parameters is messy. We'll do JSON via requests.post with params? No.
        # We'll call with query-style using requests.post and "params" for dataset_id/dataset_path and json list.
        r = requests.post(
            f"{BACKEND}/run",
            params={"dataset_id": dataset_info["dataset_id"], "dataset_path": dataset_info["path"]},
            json=opts
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            job_id = r.json()["job_id"]
            st.session_state["job_id"] = job_id
            st.success(f"Started job ✅ job_id={job_id}")

job_id = st.session_state.get("job_id")
if job_id:
    st.header("3) Status")
    st.write("Job ID:", job_id)

    if st.button("Refresh status"):
        pass

    # auto-poll a bit
    for _ in range(30):
        s = requests.get(f"{BACKEND}/status/{job_id}").json()
        st.write(s)
        if s["status"] in ["done", "failed"]:
            break
        time.sleep(2)

    st.header("4) Results")
    res = requests.get(f"{BACKEND}/results/{job_id}").json()
    st.json(res)

    st.info("ملاحظة: النتائج محفوظة على جهازك داخل outputs/<job_id>/ على نفس جهاز الـ backend.")
