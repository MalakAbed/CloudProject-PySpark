import argparse
import json
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="ML Job 3: KMeans clustering for taxi trips")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters (k)")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("TLC-KMeans").getOrCreate()
    t0 = datetime.now(timezone.utc)

    df = spark.read.parquet(args.input)

    # Trip duration in minutes (timestamp_ntz safe)
    pickup_sec = F.unix_timestamp(F.col("tpep_pickup_datetime"))
    dropoff_sec = F.unix_timestamp(F.col("tpep_dropoff_datetime"))
    df = df.withColumn("trip_duration_min", (dropoff_sec - pickup_sec) / 60.0)

    features = ["trip_distance", "trip_duration_min", "fare_amount", "tip_amount", "passenger_count"]

    clean = (
        df.select(*(features))
        .dropna()
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("trip_duration_min") > 0)
        .filter(F.col("trip_duration_min") < 300)
        .filter(F.col("trip_distance") < 200)
        .filter(F.col("fare_amount") > 0)
        .filter(F.col("tip_amount") >= 0)
        .filter(F.col("passenger_count") > 0)
        .filter(F.col("passenger_count") < 10)
    )

    # Assemble + scale features (important for KMeans)
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    vec = assembler.transform(clean)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(vec)
    vec_scaled = scaler_model.transform(vec)

    # Train KMeans
    kmeans = KMeans(k=args.k, seed=42, featuresCol="features")
    model = kmeans.fit(vec_scaled)

    preds = model.transform(vec_scaled)

    # Evaluate with silhouette score
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
    silhouette = evaluator.evaluate(preds)

    # Cluster sizes
    cluster_sizes = (
        preds.groupBy("prediction")
        .count()
        .orderBy(F.col("count").desc())
        .collect()
    )
    cluster_sizes_out = [{"cluster": int(r["prediction"]), "count": int(r["count"])} for r in cluster_sizes]

    # Cluster centers (these are in scaled space; still useful)
    centers = model.clusterCenters()
    centers_out = [c.tolist() for c in centers]

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    # Save sample with cluster assignment (for UI/report)
    preds.select(*features, F.col("prediction").alias("cluster")).limit(200).write.mode("overwrite").parquet(
        os.path.join(args.output, "clustered_sample.parquet")
    )

    result = {
        "generated_utc": utc_now(),
        "model": "KMeans",
        "k": args.k,
        "features": features,
        "rows_used": int(clean.count()),
        "silhouette": float(silhouette),
        "cluster_sizes": cluster_sizes_out,
        "cluster_centers_scaled": centers_out,
        "note": "Centers are in standardized (scaled) feature space due to StandardScaler."
    }

    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("✅ ML Job 3 (KMeans) completed")
    print(f"k={args.k}, silhouette={silhouette:.4f}")
    print(f"Output: {args.output}")
    print(f"Duration(s): {(datetime.now(timezone.utc) - t0).total_seconds():.2f}")

    spark.stop()


if __name__ == "__main__":
    main()
