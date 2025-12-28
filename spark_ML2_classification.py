import argparse
import json
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="ML Job 2: Classification (predict payment_type: card vs cash)")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("TLC-Classification").getOrCreate()
    t0 = datetime.now(timezone.utc)

    df = spark.read.parquet(args.input)

    # ---- Feature engineering: trip duration in minutes (timestamp_ntz safe) ----
    pickup_sec = F.unix_timestamp(F.col("tpep_pickup_datetime"))
    dropoff_sec = F.unix_timestamp(F.col("tpep_dropoff_datetime"))
    df = df.withColumn("trip_duration_min", (dropoff_sec - pickup_sec) / 60.0)

    # ---- Label engineering: payment_type (1=card, 2=cash) -> label (1 for card, 0 for cash) ----
    df = df.withColumn(
        "label",
        F.when(F.col("payment_type") == 1, F.lit(1.0))
         .when(F.col("payment_type") == 2, F.lit(0.0))
         .otherwise(F.lit(None))
    )

    features = ["trip_distance", "passenger_count", "trip_duration_min", "fare_amount"]

    clean = (
        df.select(*(features + ["label"]))
        .dropna()
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("trip_duration_min") > 0)
        .filter(F.col("trip_duration_min") < 300)
        .filter(F.col("trip_distance") < 200)
        .filter(F.col("fare_amount") > 0)
    )

    # Optional: show class balance (good for your report)
    class_counts = clean.groupBy("label").count().orderBy("label").collect()
    class_balance = {float(r["label"]): int(r["count"]) for r in class_counts}

    train, test = clean.randomSplit([0.8, 0.2], seed=42)

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_vec = assembler.transform(train).select("features", "label")
    test_vec = assembler.transform(test).select("features", "label")

    # Logistic Regression classifier
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
    model = lr.fit(train_vec)

    preds = model.transform(test_vec)

    # Metrics
    accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(preds)
    f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(preds)

    # Confusion matrix counts (simple + report-friendly)
    conf = (
        preds.select(
            F.col("label").cast("int").alias("label"),
            F.col("prediction").cast("int").alias("prediction")
        )
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .collect()
    )
    confusion = [{"label": int(r["label"]), "prediction": int(r["prediction"]), "count": int(r["count"])} for r in conf]

    # Write outputs
    os.makedirs(args.output, exist_ok=True)

    preds.select("label", "prediction", "probability").limit(50).write.mode("overwrite").parquet(
        os.path.join(args.output, "predictions_sample.parquet")
    )

    result = {
        "generated_utc": utc_now(),
        "model": "LogisticRegression",
        "features": features,
        "label_definition": {"1.0": "Card", "0.0": "Cash"},
        "train_rows": int(train_vec.count()),
        "test_rows": int(test_vec.count()),
        "class_balance": class_balance,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "confusion_matrix_counts": confusion,
        "coefficients": [float(x) for x in model.coefficients],
        "intercept": float(model.intercept),
    }

    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("✅ ML Job 2 (Classification) completed")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(f"Output: {args.output}")
    print(f"Duration(s): {(datetime.now(timezone.utc) - t0).total_seconds():.2f}")

    spark.stop()


if __name__ == "__main__":
    main()
