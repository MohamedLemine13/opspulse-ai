# OpsPulse AI

An intelligent IT incident detection and troubleshooting
platform built progressively using Data Engineering,
Machine Learning, Big Data and Generative AI.

## Project Objective

OpsPulse AI aims to detect abnormal behavior in IT
infrastructure by analyzing application and system
monitoring data.

The project starts with a rule-based detection baseline
and will progressively integrate more advanced AI
and data processing techniques.

## Current Stage

Stage 4: Machine Learning Incident Detection

Completed stages:

- Stage 1 — Rule-Based Incident Detection
- Stage 2 — Data Engineering Pipeline
- Stage 3 — Big Data Processing with Apache Spark

### Features

- Synthetic monitoring data generation
- Reproducible datasets
- Rule-based incident detection
- Severity classification
- Performance evaluation
- False positive and false negative analysis
- Automated unit tests

## Architecture

```text
Synthetic Data Generator
          |
          v
  sample_events.json
          |
          v
 Rule-Based Detector
          |
          v
   Incident Predictions
          |
          v
   Evaluation Metrics
```

## Technologies

- Python
- JSON
- Pytest
- Git

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate the dataset:

```bash
python -m src.generate_data
```

Run incident detection:

```bash
python -m src.detector
```

Evaluate performance:

```bash
python -m src.evaluate
```

Run automated tests:

```bash
python -m pytest tests/ -v
```

## Detection Rules

| Severity | Condition |
|----------|-----------|
| CRITICAL | Error rate >= 20% AND latency >= 1500 ms |
| WARNING | Error rate >= 10% OR latency >= 1000 ms |
| NORMAL | Otherwise |

## Expected Baseline Results

Results for the reproducible synthetic dataset
containing 50 monitoring events:

| Metric | Value |
|--------|-------|
| Precision | 80% |
| Recall | 80% |
| Accuracy | 80% |
| F1-score | 80% |

## Current Limitations

- Fixed detection thresholds
- Synthetic rather than production data
- No historical behavior analysis
- No service-specific thresholds
- CPU utilization is not considered by the detector
- No real-time data processing
- No trained Machine Learning model

## Stage 2 — Data Engineering Pipeline

The second stage introduces an ETL pipeline for
processing raw IT monitoring data and preparing
it for future Machine Learning workloads.

### Technologies

- Python
- Pandas
- NumPy
- Apache Parquet
- PyArrow
- Jupyter Notebook
- Matplotlib
- Pytest

### Pipeline Architecture

```text
Synthetic Monitoring Data
           |
           v
      JSONL Ingestion
           |
           v
      Data Validation
           |
    +------+------+
    |             |
    v             v
 Valid Data    Rejected Data
    |             |
    v             v
 Cleaning     rejected.csv
    |
    v
 Feature Engineering
    |
    v
 events.parquet
    |
    v
 Baseline Evaluation / Future ML
```

### Data Validation

The pipeline validates:

- Required fields
- Missing values
- Numeric data types
- CPU and memory utilization ranges
- Timestamp validity
- Request count consistency
- Incident labels
- Duplicate event identifiers
- JSON record integrity

Rejected records are retained with their original
content and detailed rejection reasons.

### Feature Engineering

The following features are generated:

| Feature | Description |
|---------|-------------|
| error_rate | Failed requests divided by total requests |
| no_traffic | Indicates zero-request observations |
| hour_utc | UTC hour extracted from the timestamp |

### Generate the raw dataset

```bash
python -m src.generate_raw_data
```

### Execute the preparation pipeline

```bash
python -m src.prepare_data
```

### Evaluate the rule-based baseline

```bash
python -m src.evaluate_processed
```

### Explore the dataset

```bash
jupyter notebook
```

Open `notebooks/data_exploration.ipynb`.

### Expected Data Quality Results

| Metric | Value |
|--------|-------|
| Raw records | 1020 |
| Accepted records | 1000 |
| Rejected records | 20 |
| Acceptance rate | 98.04% |
| Normal events | 700 |
| Incident events | 300 |
| Remaining missing values | 0 |
| Remaining duplicate IDs | 0 |

These results are based on a reproducible
synthetic dataset with deliberately injected
data-quality problems.

### Output Files

- `data/processed/events.parquet`
- `data/processed/rejected.csv`
- `reports/data_quality.json`

### Automated Testing

```bash
python -m pytest tests/ -v
```

The test suite covers incident detection, data
validation, feature engineering, data ingestion,
duplicate handling and pipeline reproducibility.

### Current Limitations

- Monitoring data is synthetic.
- Processing is performed on a single machine.
- Validated records are accumulated in memory.
- The data does not represent continuous,
  realistic per-service incident sequences.
- No trained Machine Learning model is integrated.
- No real-time data ingestion is implemented.

Distributed data processing will be introduced
in the next stage using Apache Spark.

## Stage 3 — Big Data Processing with Apache Spark

This stage extends the data engineering pipeline
with distributed-capable processing using PySpark.

The pipeline runs locally and introduces Spark
DataFrames, schema validation, parallel transformations,
temporal feature engineering and analytical aggregations.

### Technologies

- Apache Spark 3.5.8
- PySpark
- Spark SQL
- OpenJDK 17
- Apache Parquet
- Pandas
- NumPy
- PyArrow
- Pytest

### Pipeline Architecture

```text
Stage 2 Parquet Dataset
          |
          v
   Schema Validation
          |
          v
   Record Validation
          |
          v
   Duplicate Handling
          |
          v
   Feature Engineering
          |
    +-----+------+
    |            |
    v            v
 Service       Hourly
 Statistics    Statistics
    |            |
    +-----+------+
          |
          v
 Partitioned Parquet
```

### Spark Configuration

- Local execution with four worker threads
- 4 GB driver heap configuration
- Eight initial shuffle partitions
- Adaptive Query Execution enabled
- UTC timestamp processing

### Temporal Feature Engineering

| Feature | Description |
|---------|-------------|
| event_date | Date used for storage partitioning |
| day_of_week | Day of the week |
| is_weekend | Indicates weekend observations |
| event_hour | Timestamp truncated to the hour |

### Analytical Aggregations

The pipeline generates:

- Service-level monitoring statistics
- Hourly monitoring statistics per service
- Incident counts and incident rates
- Average CPU and memory utilization
- Average response latency
- Observation-level and global error rates

### Data Validation

The pipeline checks the incoming schema,
numeric ranges, missing values, request consistency
and incident labels.

All records associated with duplicate event IDs
are rejected to ensure deterministic behavior.

### Storage Optimization

Processed events are stored in Apache Parquet
with date-based directory partitioning.

Spark execution plans are inspected to understand:

- Lazy evaluation
- Predicate pushdown
- Column pruning
- Partition pruning
- Shuffle operations
- Adaptive Query Execution

### Installation

Install OpenJDK 17 and configure JAVA_HOME.

```bash
pip install -r requirements.txt
```

### Verify Spark Configuration

```bash
python -m src.spark_config
```

### Run the Spark Pipeline

Generate the Stage 2 dataset first:

```bash
python -m src.generate_raw_data
python -m src.prepare_data
```

Execute the Spark pipeline:

```bash
python -m src.spark_pipeline
```

### Generate Benchmark Datasets

```bash
python -m src.generate_large_data
```

Three synthetic datasets are generated:

- 10,000 observations
- 100,000 observations
- 1,000,000 observations

### Process a Larger Dataset

```bash
python -m src.spark_pipeline \
    --input data/large/events_1000000.parquet
```

### Performance Benchmark

```bash
python -m src.benchmark
```

The benchmark compares equivalent Pandas and Spark
operations using the same Parquet input datasets.

Execution times are measured across repeated runs,
with Spark startup measured separately.

Output equivalence is verified before performance
results are accepted.

### Benchmark Results

| Records | Pandas (s) | Spark (s) | Spark Speedup |
|---------|------------|-----------|---------------|
| 10,000 | TBD | TBD | TBD |
| 100,000 | TBD | TBD | TBD |
| 1,000,000 | TBD | TBD | TBD |

Results must be measured on the local machine.

### Output Files

- `data/spark/events/`
- `data/spark/service_statistics/`
- `data/spark/hourly_statistics/`
- `data/spark/rejected/`
- `reports/spark_quality.json`
- `reports/benchmark_results.csv`

### Benchmark Visualization

Open `notebooks/spark_benchmark.ipynb`
to visualize the execution times and storage sizes.

### Automated Testing

```bash
python -m pytest tests/ -v
```

Tests cover schema validation, feature engineering,
duplicate handling, distributed-style aggregations,
Parquet output and Pandas/Spark result equivalence.

### Current Limitations

- Spark is executed locally rather than across
  a multi-machine cluster.
- All monitoring datasets are synthetic.
- The larger benchmarking datasets are derived
  from the original Stage 2 dataset.
- Memory consumption is not benchmarked.
- No real-time Kafka integration is implemented.
- No Machine Learning model has been trained yet.

## Stage 4 — Machine Learning Incident Detection

This stage introduces supervised Machine Learning
and unsupervised anomaly detection to identify
incidents from IT infrastructure monitoring data.

### Technologies

- Scikit-learn
- Pandas
- NumPy
- Joblib
- Matplotlib
- Pytest

### Models

| Model | Learning Type |
|-------|---------------|
| Rule-based detector | Baseline |
| Logistic Regression | Supervised |
| Random Forest | Supervised |
| Isolation Forest | Unsupervised |

### Input Features

- CPU utilization
- Memory utilization
- Error rate
- P95 response latency

The target variable is `incident_label`,
where 0 represents normal behavior and
1 represents an incident.

### Dataset Strategy

The original validated Stage 2 dataset is used.

Chronological splitting is performed without shuffling:

| Split | Proportion | Observations |
|-------|------------|--------------|
| Training | 70% | 700 |
| Validation | 15% | 150 |
| Testing | 15% | 150 |

The resampled Stage 3 benchmarking datasets
are excluded from model training to reduce
the risk of evaluation leakage.

### Training Architecture

```text
Validated Monitoring Data
           |
           v
    Feature Selection
           |
           v
   Chronological Split
           |
           v
     Model Training
           |
           v
   Validation Comparison
           |
           v
     Model Selection
           |
           v
   Held-out Evaluation
           |
           v
  Model and Metadata
           |
           v
        Inference
```

### Model Training

Generate and prepare the dataset:

```bash
python -m src.generate_raw_data
python -m src.prepare_data
```

Train and evaluate the models:

```bash
python -m src.ml.train
```

Model selection is based on validation F1-score.

Only the selected ML candidate and the original
rule-based baseline are evaluated on the held-out
test dataset.

### Validation Results

| Model | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Rule-based | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| Isolation Forest | TBD | TBD | TBD |

### Held-out Test Results

| Model | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Rule-based | TBD | TBD | TBD |
| Selected ML Model | TBD | TBD | TBD |

Populate these tables using measured results.

### Model Persistence

The selected Machine Learning model is saved as:

- `models/incident_model.joblib`
- `models/model_metadata.json`

Metadata contains model information, feature
definitions, dataset fingerprints, library versions,
split statistics and evaluation results.

Model artifacts must be regenerated locally.

Only trusted Joblib artifacts should be loaded.

### Incident Inference

```bash
python -m src.ml.predict \
    --cpu 98 \
    --memory 70 \
    --error-rate 0.03 \
    --latency 500
```

The inference component validates incoming features
and returns a binary incident prediction.

Supervised classifiers additionally provide an
estimated incident probability.

Isolation Forest provides an anomaly score instead.

### Evaluation Reports

- `reports/ml/dataset_summary.json`
- `reports/ml/validation_metrics.csv`
- `reports/ml/test_metrics.csv`
- `reports/ml/test_predictions.csv`

### Exploratory ML Analysis

Open:

`notebooks/ml_experiments.ipynb`

The notebook contains model comparisons,
confusion matrices, error analysis and
model interpretation.

### Automated Testing

```bash
python -m pytest tests/ -v
```

The tests cover feature validation,
chronological splitting, model training,
evaluation, serialization and inference.

### Current Limitations

- Training data is synthetic.
- The dataset contains only 1,000 observations.
- Simulated operational scenarios may be easier
  to distinguish than real incidents.
- Chronological splitting does not reproduce
  realistic production drift.
- Model hyperparameters are fixed for the
  initial experiment.
- Estimated probabilities have not been calibrated.
- No production model deployment is implemented.

Further model evaluation and optimization
will be introduced in the next stage.

## Roadmap

- [x] Rule-based incident detection
- [x] Data preparation and validation pipeline
- [x] Big Data processing with Apache Spark
- [x] Machine Learning-based incident detection
- [ ] Model evaluation and optimization
- [ ] Deep Learning-based log analysis
- [ ] Generative AI and RAG troubleshooting
- [ ] FastAPI integration
- [ ] Kafka and Spark Streaming
- [ ] Docker and MLOps deployment

## Project Status

Under active development.