import subprocess
import os

DATA = "/Users/malakabed/Desktop/CloudProject/data/yellow_tripdata_2025-01.parquet"
BASE_OUT = "./simulation_results"

os.makedirs(BASE_OUT, exist_ok=True)

workers_list = [1, 2, 4, 8]

for w in workers_list:
    out_dir = f"{BASE_OUT}/local_{w}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python3",
        "spark_stats_basic.py",
        "--input", DATA,
        "--output", out_dir,
        "--workers", str(w)
    ]

    print(f"\nRunning with local[{w}] ...")
    subprocess.run(cmd, check=True)
