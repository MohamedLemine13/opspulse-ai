from pyspark.sql import SparkSession


def create_spark_session(
    app_name: str = "OpsPulse AI",
    master: str = "local[4]",
) -> SparkSession:
    """Create and configure a local Spark session."""

    spark = (
        SparkSession.builder

        .appName(app_name)

        .master(master)

        .config(
            "spark.driver.memory",
            "4g",
        )

        .config(
            "spark.sql.shuffle.partitions",
            "8",
        )

        .config(
            "spark.sql.session.timeZone",
            "UTC",
        )

        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )

        .config(
            "spark.driver.host",
            "127.0.0.1",
        )

        .config(
            "spark.driver.bindAddress",
            "127.0.0.1",
        )

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


def main():

    spark = create_spark_session()

    try:

        print("=" * 40)
        print("OPSPULSE AI - SPARK CONFIGURATION")
        print("=" * 40)

        print(f"Spark version: {spark.version}")

        print(
            f"Master: "
            f"{spark.sparkContext.master}"
        )

        print(
            f"Default parallelism: "
            f"{spark.sparkContext.defaultParallelism}"
        )

        print(
            f"Shuffle partitions: "
            f"{spark.conf.get('spark.sql.shuffle.partitions')}"
        )

        df = spark.range(10)

        df.show()

    finally:

        spark.stop()


if __name__ == "__main__":
    main()