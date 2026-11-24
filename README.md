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

Stage 2: Data Engineering Pipeline

Stage 1 — Rule-Based Incident Detection is completed.

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

## Roadmap

- [x] Rule-based incident detection
- [x] Data preparation and validation pipeline
- [ ] Big Data processing with Apache Spark
- [ ] Machine Learning-based incident detection
- [ ] Model evaluation and benchmarking
- [ ] Deep Learning-based log analysis
- [ ] Generative AI and RAG troubleshooting
- [ ] FastAPI integration
- [ ] Kafka and Spark Streaming
- [ ] Docker and MLOps deployment

## Project Status

Under active development.