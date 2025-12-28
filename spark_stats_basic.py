import argparse
import json
import os
import time
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Basic descriptive stats for TLC Parquet")
    parser.add_argument("--input", required=True, help="Input Parquet path (file or folder)")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--workers", type=int, default=1, help="Number of local workers")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName(f"Stats_local_{args.workers}")
        .master(f"local[{args.workers}]")
        .getOrCreate()
    )

    start_time = time.time()

    df = spark.read.parquet(args.input)

    # 1) Row count
    row_count = df.count()

    # 2) Column count + schema
    col_count = len(df.columns)
    schema_info = [{"name": f.name, "type": f.dataType.simpleString()} for f in df.schema.fields]

    # 3) Null counts
    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
    null_row = df.select(*null_exprs).collect()[0].asDict()
    null_counts = [{"column": k, "null_count": int(v)} for k, v in null_row.items()]
    null_counts_sorted = sorted(null_counts, key=lambda x: x["null_count"], reverse=True)

    # 4) Numeric summary (if columns exist)
    candidate_numeric_cols = ["trip_distance", "fare_amount", "total_amount", "tip_amount"]
    present_numeric_cols = [c for c in candidate_numeric_cols if c in df.columns]

    numeric_summary = []
    if present_numeric_cols:
        agg_exprs = []
        for c in present_numeric_cols:
            agg_exprs += [
                F.min(F.col(c)).alias(f"{c}__min"),
                F.max(F.col(c)).alias(f"{c}__max"),
                F.avg(F.col(c)).alias(f"{c}__mean"),
            ]
        agg = df.agg(*agg_exprs).collect()[0].asDict()
        for c in present_numeric_cols:
            numeric_summary.append(
                {
                    "column": c,
                    "min": agg.get(f"{c}__min"),
                    "max": agg.get(f"{c}__max"),
                    "mean": agg.get(f"{c}__mean"),
                }
            )

    # 5) Top payment_type (avoid pandas to keep it simple)
    top_payment = []
    if "payment_type" in df.columns:
        rows = (
            df.groupBy("payment_type")
            .count()
            .orderBy(F.col("count").desc())
            .limit(10)
            .collect()
        )
        top_payment = [{"payment_type": r["payment_type"], "count": int(r["count"])} for r in rows]

    # Ensure output folder exists BEFORE writing files
    os.makedirs(args.output, exist_ok=True)

    # Save a small preview
    df.limit(50).write.mode("overwrite").parquet(os.path.join(args.output, "preview_50rows.parquet"))

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    # Save runtime
    with open(os.path.join(args.output, "runtime.txt"), "w") as f:
        f.write(str(duration))

    summary = {
        "generated_utc": utc_now_iso(),
        "input": args.input,
        "workers": args.workers,
        "row_count": row_count,
        "column_count": col_count,
        "duration_seconds": duration,
        "schema": schema_info,
        "numeric_summary": numeric_summary,
        "top_null_columns_preview": null_counts_sorted[:20],
        "top_payment_type_preview": top_payment,
    }

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(args.output, "null_counts.json"), "w", encoding="utf-8") as f:
        json.dump(null_counts_sorted, f, indent=2)

    print("✅ DONE")
    print(f"Workers: {args.workers}")
    print(f"Rows: {row_count:,}")
    print(f"Cols: {col_count}")
    print(f"Runtime (sec): {duration}")
    print(f"Output folder: {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
