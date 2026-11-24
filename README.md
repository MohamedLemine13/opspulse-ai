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

Stage 1: Rule-Based Incident Detection

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

## Roadmap

- [x] Rule-based incident detection
- [ ] Data preparation and validation pipeline
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