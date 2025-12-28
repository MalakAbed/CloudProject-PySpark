import argparse
import json
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="ML Job 1: Regression (predict total_amount)")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("TLC-Regression").getOrCreate()
    t0 = datetime.now(timezone.utc)

    df = spark.read.parquet(args.input)

    pickup_sec = F.unix_timestamp(F.col("tpep_pickup_datetime"))
    dropoff_sec = F.unix_timestamp(F.col("tpep_dropoff_datetime"))
    df = df.withColumn("trip_duration_min", (dropoff_sec - pickup_sec) / 60.0)

    features = ["trip_distance", "passenger_count", "trip_duration_min"]
    label = "total_amount"

    clean = (
        df.select(*(features + [label]))
        .dropna()
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("trip_duration_min") > 0)
        .filter(F.col("trip_duration_min") < 300)
        .filter(F.col("trip_distance") < 200)
        .filter(F.col(label) > 0)
    )

    train, test = clean.randomSplit([0.8, 0.2], seed=42)

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_vec = assembler.transform(train).select("features", F.col(label).alias("label"))
    test_vec = assembler.transform(test).select("features", F.col(label).alias("label"))

    lr = LinearRegression()
    model = lr.fit(train_vec)

    predictions = model.transform(test_vec)

    rmse = RegressionEvaluator(metricName="rmse").evaluate(predictions)
    r2 = RegressionEvaluator(metricName="r2").evaluate(predictions)

    os.makedirs(args.output, exist_ok=True)

    predictions.select("label", "prediction").limit(50).write.mode("overwrite").parquet(
        os.path.join(args.output, "predictions_sample.parquet")
    )

    result = {
        "generated_utc": utc_now(),
        "model": "LinearRegression",
        "features": features,
        "label": label,
        "train_rows": train_vec.count(),
        "test_rows": test_vec.count(),
        "rmse": float(rmse),
        "r2": float(r2),
        "coefficients": [float(x) for x in model.coefficients],
        "intercept": float(model.intercept),
    }

    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("✅ ML Job 1 (Regression) completed")
    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}")
    print(f"Output: {args.output}")
    print(f"Duration(s): {(datetime.now(timezone.utc) - t0).total_seconds():.2f}")

    spark.stop()


if __name__ == "__main__":
    main()
