# Software Engineer (ML & LLMs) Challenge

## 1. Overview

This project operationalizes a machine learning solution into a production-ready system capable of predicting flight delays for SCL airport operations.

The solution includes:

- A machine learning prediction pipeline
- A preprocessing and inference workflow
- A REST API built with FastAPI
- Cloud deployment using Google Cloud Run
- Stress testing to validate reliability under load
- Stress testing to validate reliability under load

The objective of the challenge was to transform a data science notebook into a maintainable and deployable production system.

---

# 2. Problem Understanding

The dataset contains operational and temporal flight information such as:

- Airline (`OPERA`)
- Flight type (`TIPOVUELO`)
- Month of operation (`MES`)
- Flight timestamps and operational metadata

The target variable is defined as:

> `delay = 1 if min_diff > 15 minutes, otherwise 0`

This converts the problem into a binary classification task where the system predicts whether a flight will experience delay.

---

# 3. Machine Learning Pipeline

## 3.1 Model Selection and Experiments

Different models proposed in the notebook were evaluated, including Logistic Regression and XGBoost.

Logistic Regression showed stable results across different hyperparameter configurations while achieving performance comparable to more complex models such as XGBoost. Since the dataset is imbalanced, the evaluation process focused primarily on the minority class (delayed flights), especially recall and F1-score for class 1.

The experiments demonstrated that applying class balancing techniques improved the model more significantly than extensive hyperparameter tuning. In contrast, tuning produced only marginal improvements while increasing model complexity and operational overhead.

For this reason, the final model selected for production was a **balanced Logistic Regression**, prioritizing:

- Stability
- Simplicity
- Interpretability
- Ease of maintenance
- Fast inference time

Additionally, Logistic Regression demonstrated highly stable behavior across different parameter values while maintaining competitive performance compared to XGBoost. Because the performance gains from more complex models were minimal, Logistic Regression was selected as the production model due to its lower operational complexity and easier explainability.

---

## 3.2 Python and Environment Stability

During development, Python 3.10 was selected as the most stable runtime version for the project.

This decision was made because newer Python versions introduced compatibility issues with several dependencies used in the challenge environment, including:

- `numpy`
- `pandas`
- `scikit-learn`
- `gevent`
- `locust`

Using Python 3.10 allowed the project to remain stable without introducing unnecessary dependency upgrades or modifications to the libraries originally provided in the challenge.

This ensured reproducibility across:
- Local development
- Testing
- Docker execution
- Cloud deployment

---

## 3.3 Preprocessing Pipeline

A key aspect of the implementation was designing a preprocessing pipeline that behaves consistently during both training and inference.

During the operationalization process, it became necessary to ensure that the transformations applied to the training dataset were reproduced identically when serving predictions through the API. This was especially important because the model relies on engineered categorical features that must maintain the same structure during inference.

To address this, preprocessing logic was centralized inside the `DelayModel` implementation. The final preprocessing pipeline guarantees that:

- The same transformations applied during training are replicated during inference
- Feature columns remain aligned through a fixed `TOP_FEATURES` structure
- Missing categorical variables are consistently handled
- Prediction inputs always match the expected feature space

---

## 3.4 Feature Engineering

The preprocessing stage includes transformations such as:

- One-hot encoding of categorical variables
- Airline feature encoding (`OPERA_*`)
- Flight type encoding (`TIPOVUELO_*`)
- Month encoding (`MES_*`)
- Feature alignment using `TOP_FEATURES`

This ensures compatibility between:
- training data
- API requests
- deployed inference environment

---

## 3.5 Final Design Decision

The final architecture prioritizes:

- Stability over unnecessary complexity
- Consistent preprocessing
- Lightweight inference
- Maintainability
- Reproducibility across environments

The solution was intentionally designed to remain simple, robust, and production-friendly.

---

# 4. API Design (FastAPI)

## 4.1 Architecture

The API was implemented using FastAPI and exposes a REST interface for prediction requests.

The API includes:

- Pydantic validation
- Structured request schemas
- Model preprocessing
- Prediction inference
- JSON responses

---

## 4.2 Endpoints

### Health Check

```http
GET /health

Response:

{
  "status": "OK"
}

This endpoint is used to verify service availability.

Prediction Endpoint
POST /predict

Example request:

{
  "flights": [
    {
      "OPERA": "Grupo LATAM",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}

Example response:

{
  "predict": [0]
}
4.3 Input Validation

Validation rules were implemented using Pydantic and custom validation logic.

Rules include:

MES must be between 1 and 12
TIPOVUELO must be either:
N
I

Invalid requests return:

HTTP 400 Bad Request

This ensures input integrity before model execution.

5. Testing
5.1 Model Tests

The model implementation was validated using the provided test suite.

Several issues had to be resolved during testing, including:

Dataset path inconsistencies
Relative path execution problems
Feature alignment issues between preprocessing stages

The project was updated to ensure tests execute correctly regardless of execution context.

5.2 API Tests

The FastAPI implementation successfully passed the API test suite.

Tests validated:

Endpoint availability
Correct response structure
Validation behavior
Prediction execution
6. Deployment (Google Cloud Platform)
6.1 Deployment Strategy

The API was deployed using Google Cloud Run as a serverless containerized service.

The deployment process included:

Containerization with Docker
FastAPI execution through Uvicorn
Cloud Run deployment
Public HTTPS exposure
6.2 GCP IAM and Permission Issues

During deployment, several IAM permission issues were encountered.

The initial deployment failed because the default service account lacked the required permissions for Cloud Build and Cloud Run operations.

The following issues had to be resolved:

Missing IAM bindings
Incorrect service account configuration
Cloud Build permission restrictions
Deployment authorization failures

These were solved by:

identifying the correct service account
assigning the appropriate IAM roles
configuring Cloud Build permissions correctly

Once permissions were updated, deployment completed successfully.

This reflects a common real-world cloud engineering challenge when deploying production systems.

6.3 Production URL

Final deployed API:

https://ml-api-805059731847.us-central1.run.app
7. Stress Testing
7.1 Goal

The objective of the stress test was to validate that the deployed API could handle repeated requests under load without failures.

7.2 Locust and Environment Challenges

The challenge originally provided a Locust-based stress test.

However, several environment-related issues occurred during execution, including:

WSL integration problems
Docker integration issues
gevent installation failures
Python dependency conflicts
PEP 668 externally-managed environment restrictions

Additional complexity arose from differences between:

Git Bash
Windows Python
WSL environments
7.3 Alternative Stress Testing Strategy

To ensure successful validation of the deployed API, a lightweight Python-based stress testing script using requests was implemented.

The test simulated:

Multiple sequential POST requests
Alternating airline payloads
Continuous API interaction

The deployed endpoint was tested directly through Google Cloud Run.

7.4 Stress Test Results

Results:

200 successful requests
HTTP 200 responses throughout execution
No failed predictions
Stable API behavior under repeated load

This confirmed that the deployed service behaves reliably under stress conditions.

8. Environment and Debugging Challenges

Several real-world engineering issues were encountered and resolved during development:

8.1 WSL vs Windows Filesystem

Problems occurred when executing environments inside /mnt/c, particularly related to:

permissions
symlinks
virtual environments

The solution was to execute the project inside the native Linux filesystem under /home.

8.2 Virtual Environment and Package Management

Ubuntu's externally-managed Python environment (PEP 668) restricted direct package installation through pip.

The solution involved:

isolated virtual environments
environment-specific dependency installation
separation of development and testing contexts
8.3 Docker Integration

Docker Desktop integration with WSL initially failed, preventing Locust container execution.

As a result:

stress testing was migrated to a Python-based implementation
deployment continued independently through GCP Cloud Run
9. Final Architecture
Client
  ↓
FastAPI (/predict)
  ↓
Pydantic Validation
  ↓
Preprocessing Pipeline (DelayModel)
  ↓
Machine Learning Inference
  ↓
JSON Response
  ↓
Google Cloud Run Deployment
10. Conclusion

This project successfully transformed a data science notebook into a production-ready machine learning system.

Key accomplishments include:

End-to-end ML operationalization
Stable preprocessing and inference pipeline
Production-ready FastAPI implementation
Cloud deployment using Google Cloud Run
Successful stress testing under load
Resolution of real-world engineering challenges involving:
cloud permissions
dependency management
environment stability
deployment debugging
testing reproducibility

The final system is stable, maintainable, reproducible, and ready for production-style evaluation.
