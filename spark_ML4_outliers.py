import argparse
import json
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="ML Job 4: Outlier detection using z-score")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--z", type=float, default=3.0, help="Z-score threshold")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("TLC-Outliers").getOrCreate()
    t0 = datetime.now(timezone.utc)

    df = spark.read.parquet(args.input)

    # Trip duration in minutes (timestamp_ntz safe)
    pickup_sec = F.unix_timestamp(F.col("tpep_pickup_datetime"))
    dropoff_sec = F.unix_timestamp(F.col("tpep_dropoff_datetime"))
    df = df.withColumn("trip_duration_min", (dropoff_sec - pickup_sec) / 60.0)

    features = ["trip_distance", "total_amount", "trip_duration_min"]

    clean = (
        df.select(*(features))
        .dropna()
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("trip_duration_min") > 0)
        .filter(F.col("trip_duration_min") < 300)
        .filter(F.col("trip_distance") < 200)
        .filter(F.col("total_amount") > 0)
    )

    # Compute mean and std for each feature
    stats = clean.agg(
        *[
            F.mean(c).alias(f"{c}_mean") for c in features
        ],
        *[
            F.stddev(c).alias(f"{c}_std") for c in features
        ]
    ).collect()[0]

    means = {c: float(stats[f"{c}_mean"]) for c in features}
    stds = {c: float(stats[f"{c}_std"]) for c in features}

    # Add z-score columns
    for c in features:
        df = df.withColumn(f"{c}_z", (F.col(c) - means[c]) / stds[c])

    # Flag outliers
    outlier_cond = None
    for c in features:
        cond = F.abs(F.col(f"{c}_z")) > args.z
        outlier_cond = cond if outlier_cond is None else (outlier_cond | cond)

    outliers = df.filter(outlier_cond)

    total_rows = clean.count()
    outlier_count = outliers.count()
    outlier_ratio = outlier_count / total_rows if total_rows > 0 else 0.0

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    outliers.select(
        "trip_distance",
        "total_amount",
        "trip_duration_min",
        *(f"{c}_z" for c in features)
    ).limit(200).write.mode("overwrite").parquet(
        os.path.join(args.output, "outliers_sample.parquet")
    )

    result = {
        "generated_utc": utc_now(),
        "method": "z-score",
        "z_threshold": args.z,
        "features": features,
        "means": means,
        "stds": stds,
        "total_rows_used": int(total_rows),
        "outlier_count": int(outlier_count),
        "outlier_ratio": outlier_ratio,
    }

    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("✅ ML Job 4 (Outlier Detection) completed")
    print(f"Outliers: {outlier_count} / {total_rows} ({outlier_ratio:.4%})")
    print(f"Output: {args.output}")
    print(f"Duration(s): {(datetime.now(timezone.utc) - t0).total_seconds():.2f}")

    spark.stop()


if __name__ == "__main__":
    main()
