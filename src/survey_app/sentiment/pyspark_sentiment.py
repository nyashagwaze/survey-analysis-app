"""
PySpark Sentiment Analysis Module
Parallel sentiment analysis using Pandas UDFs for batch processing.
10-100x faster than pandas .iterrows() approach.
Uses improved sentiment_module with expanded coping detection.
Respects meaningful flags from null_text_detector for column-level filtering.
"""

from .sentiment_module import weighted_sentiment_for_row, DEFAULT_COLUMN_WEIGHTS

def _build_sentiment_udf():
    from typing import Iterator
    import warnings
    import os
    import pandas as pd

    try:
        from pyspark.sql.functions import pandas_udf, col
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    except Exception as exc:
        raise ImportError("pyspark is required for Spark sentiment UDFs.") from exc

    # Suppress transformers warnings
    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    SENTIMENT_SCHEMA = StructType([
        StructField("compound_weighted", DoubleType(), True),
        StructField("sentiment_label", StringType(), True),
        StructField("coping_flag", StringType(), True),
        StructField("compound_Free_Text_1", DoubleType(), True),
        StructField("compound_Free_Text_2", DoubleType(), True),
        StructField("compound_Free_Text_3", DoubleType(), True)
    ])

    @pandas_udf(SENTIMENT_SCHEMA)
    def compute_sentiment_batch(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """
        Pandas UDF for batch sentiment processing.
        Processes rows in batches for better performance.
        Uses improved coping detection with 30+ patterns.
        Respects meaningful flags to skip dismissed/null text.
        """
        text_columns = ["Free_Text_1", "Free_Text_2", "Free_Text_3"]

        for batch_df in iterator:
            results = []

            for _, row in batch_df.iterrows():
                result = weighted_sentiment_for_row(
                    row=row.to_dict(),
                    text_columns=text_columns,
                    weights=DEFAULT_COLUMN_WEIGHTS,
                    pos=0.05,
                    neg=-0.05
                )

                compound_by_col = result["compound_by_column"]

                results.append({
                    "compound_weighted": result["compound_weighted"],
                    "sentiment_label": result["sentiment_label"],
                    "coping_flag": result["coping_flag"],
                    "compound_Free_Text_1": compound_by_col.get("Free_Text_1"),
                    "compound_Free_Text_2": compound_by_col.get("Free_Text_2"),
                    "compound_Free_Text_3": compound_by_col.get("Free_Text_3")
                })

            yield pd.DataFrame(results)

    return compute_sentiment_batch, col


def add_sentiment_columns(df_spark, text_columns=["Free_Text_1", "Free_Text_2", "Free_Text_3"]):
    """
    Add sentiment analysis columns to Spark DataFrame.
    Respects meaningful flags from null_text_detector.
    """
    compute_sentiment_batch, col = _build_sentiment_udf()

    columns_for_udf = []
    for text_col in text_columns:
        columns_for_udf.append(text_col)
        meaningful_flag = f"{text_col}_is_meaningful"
        if meaningful_flag in df_spark.columns:
            columns_for_udf.append(meaningful_flag)

    df_with_sentiment = df_spark.withColumn(
        "sentiment_result",
        compute_sentiment_batch(*[col(c) for c in columns_for_udf])
    )

    df_with_sentiment = df_with_sentiment.select(
        "*",
        col("sentiment_result.compound_weighted").alias("compound"),
        col("sentiment_result.sentiment_label").alias("sentiment_label"),
        col("sentiment_result.coping_flag").alias("coping_flag"),
        col("sentiment_result.compound_Free_Text_1").alias("compound_Free_Text_1"),
        col("sentiment_result.compound_Free_Text_2").alias("compound_Free_Text_2"),
        col("sentiment_result.compound_Free_Text_3").alias("compound_Free_Text_3")
    ).drop("sentiment_result")

    return df_with_sentiment
