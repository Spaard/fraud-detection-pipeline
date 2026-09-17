# Transaction Fraud Detection Dashboard

![CI/CD Pipeline](https://github.com/Spaard/fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.13-blue)
![Docker Image](https://img.shields.io/badge/docker-ready-brightgreen)

An end-to-end MLOps and analytics application designed to detect, analyze, and visualize credit card payment fraud. Built with Streamlit, containerized with Docker, and managed using `uv`.

---

## 📌 Overview

This repository contains a full MLOps pipeline and interactive analytical dashboard designed to monitor card payment fraud:

- **Optimized Data Layer**: Loads and parses transaction datasets efficiently using Parquet storage.
- **Interactive Dashboard**: Real-time KPI tracking, Pydeck geographic risk mapping with tooltips, amount distribution analytics, ML alert threshold simulator, and suspicious transaction inspection.
- **Reproducible Setup**: Environment managed with `uv` lockfiles and containerized via multi-stage Docker builds.
- **Automated CI/CD & Registry Deployment**: Unit testing and coverage reporting on every push/PR via GitHub Actions. Every merge to `main` automatically builds and deploys the latest image to Docker Hub (`spaard/fraud-detection-app:latest`).

---

## 🛠️ Project Structure

```text
├── .github/workflows/   # CI/CD pipeline definition
├── data/               # Parquet dataset
├── src/                # Core modules and data processing
│   ├── __init__.py
│   └── data_loader.py
├── tests/              # Automated unit test suite (Pytest)
├── app.py              # Streamlit dashboard application
├── Dockerfile          # Multi-stage production Docker image
├── pyproject.toml      # Project configuration and dependencies
└── uv.lock             # Exact dependency lockfile
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites

Ensure uv is installed on your machine.

### Local Execution

Clone the repository:

```
git cl one git@github.com:Spaard/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

Run unit tests:

```
uv run pytest
```

Launch the Streamlit dashboard:

```
uv run streamlit run app.py
```

---

## 🐳 Docker Deployment

### Continuous Deployment Pipeline

Whenever new code is pushed or merged into the main branch, the GitHub Actions workflow automatically:

Executes the Pytest suite and validates code coverage.

Builds the multi-stage Docker image using uv.

Pushes the updated image to the Docker Hub Registry under the spaard/fraud-detection-app:latest tag.

### Build and Run Locally

Build the Docker image:

```
docker build -t fraud-detection-app .
```

Run the container:

```
docker run -p 8501:8501 fraud-detection-app
```

Open http://localhost:8501 in your browser.

### Pull Pre-built Image from Docker Hub

```
docker pull spaard/fraud-detection-app:latest
docker run -p 8501:8501 spaard/fraud-detection-app:latest
```

---

## 🧪 Testing & Quality Assurance

Unit tests cover data loading, filter range boundaries, edge cases (empty dataframes), and metric computations.

Run the test suite with coverage report:

```
uv run pytest --cov=src
```
